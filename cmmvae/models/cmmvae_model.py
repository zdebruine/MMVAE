from typing import Optional, Union, Dict, List, Tuple
import pandas as pd
import torch
import torch.nn as nn

from .base_model import BaseModel
from cmmvae.modules import CMMVAE
from cmmvae.constants import REGISTRY_KEYS as RK

# Gradient reversal layer function
class GradientReversalFunction(torch.autograd.Function):
    """
    Implements a gradient reversal layer (GRL) for adversarial training.

    This function inverts the gradients during the backward pass, allowing
    for adversarial objectives to be optimized in the opposite direction.

    Attributes:
        lambd (float): The scaling factor for gradient reversal.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        """
        Forward pass for gradient reversal.

        Args:
            ctx: Context object for storing information for backward computation.
            x (torch.Tensor): Input tensor.
            lambd (float): Scaling factor for gradient reversal.

        Returns:
            torch.Tensor: Output tensor, identical to input.
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass with reversed gradients.

        Args:
            ctx: Context object with stored information from forward computation.
            grad_output (torch.Tensor): Gradient of the output with respect to input.

        Returns:
            Tuple[torch.Tensor, None]: Reversed gradient for input and None for lambd.
        """
        return grad_output.neg() * ctx.lambd, None


class CMMVAEModel(BaseModel):
    """
    Conditional Multi-Modal Variational Autoencoder (CMMVAE) model for handling expert-specific data.

    This class is designed for training VAEs with multiple experts and adversarial components.

    Args:
        module (CMMVAE): Conditional Multi-Modal VAE module.
        **kwargs: Additional keyword arguments for the base VAE model.

    Attributes:
        module (CMMVAE): The CMMVAE module for processing and generating data.
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
        adversarial_criterion (nn.CrossEntropyLoss): Loss function for adversarial training.
    """
    
    module: CMMVAE
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # Disable automatic optimization for manual control
        self.adversarial_criterion = nn.CrossEntropyLoss()  # Criterion for adversarial loss
        
    def training_step(self, batch: Tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int) -> None:
        """
        Perform a single training step.

        This involves encoding the input, calculating losses, and updating weights.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        x, metadata, expert_id = batch
        expert_label = self.module.experts.labels[expert_id]

        # Retrieve optimizers
        optims = self.get_optimizers(zero_all=True)
        expert_optimizer = optims['experts'][expert_id]
        vae_optimizer = optims['vae']
        adversarial_optimizers = optims['adversarials']

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, cg_xhats, hidden_representations = self.module(x, metadata, expert_id)
        
        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        loss_dict = self.module.vae.elbo(qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight)

        # Train adversarial networks
        adversarial_loss = None
        for i, (hidden_rep, adv) in enumerate(zip(hidden_representations, self.module.adversarials)):
            # Create expert labels
            expert_labels = torch.full((self.batch_size,), expert_label, dtype=torch.long, device=x.device)

            # Apply gradient reversal
            reversed_hidden_rep = GradientReversalFunction.apply(hidden_rep, 0.1)
            
            # Get adversarial predictions
            adv_output = adv(reversed_hidden_rep)

            # Calculate adversarial loss
            current_adversarial_loss = self.adversarial_criterion(adv_output, expert_labels)
            if adversarial_loss is None:
                adversarial_loss = current_adversarial_loss
            else:
                adversarial_loss += current_adversarial_loss

            # Backpropagation for the adversarial
            self.manual_backward(current_adversarial_loss, retain_graph=True)
            adversarial_optimizers[i].step()
            
        loss_dict['adversarial_loss'] = adversarial_loss
        loss = loss_dict[RK.LOSS]
        
        # Backpropagation for encoder and decoder
        self.manual_backward(loss)

        # Clip gradients for stability
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        
        # Handle inactive conditional modules
        inactive_modules = []
        if self.module.vae.conditionals:
            for conditional_layer in self.module.vae.conditionals.layers:
                inactive = conditional_layer.unique_conditions - conditional_layer.active_condition_modules
                inactive_modules.extend([conditional_layer.conditions[key] for key in conditional_layer.conditions if key in inactive])
        
        for module in inactive_modules:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # Update the weights
        vae_optimizer.step()
        expert_optimizer.step()
        self.kl_annealing_fn.step()
        
        # Log the loss
        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])
        
    def validation_step(self, batch: Tuple[torch.Tensor, pd.DataFrame, str]) -> None:
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id = batch
        expert_label = self.module.experts.labels[expert_id]

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, cg_xhats, hidden_representations = self.module(x, metadata, expert_id)
        
        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        loss_dict = self.module.vae.elbo(qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight)

        # Calculate adversarial loss
        adversarial_loss = None
        for i, (hidden_rep, adv) in enumerate(zip(hidden_representations, self.module.adversarials)):
            # Create expert labels
            expert_labels = torch.full((self.batch_size,), expert_label, dtype=torch.long, device=x.device)

            # Apply gradient reversal
            reversed_hidden_rep = GradientReversalFunction.apply(hidden_rep, 0.1)

            # Get adversarial predictions
            adv_output = adv(reversed_hidden_rep)

            # Calculate adversarial loss
            current_adversarial_loss = self.adversarial_criterion(adv_output, expert_labels)
            loss_dict[f'adv_{i}'] = current_adversarial_loss
            
            if adversarial_loss is None:
                adversarial_loss = current_adversarial_loss
            else:
                adversarial_loss += current_adversarial_loss
            
        loss_dict[RK.LOSS] += adversarial_loss
        
        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])
        
    # Alias for validation_step method to reuse for testing
    test_step = validation_step
    
    def predict_step(self, batch: Tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int) -> None:
        """
        Perform a prediction step.

        This step extracts latent embeddings and saves them for analysis.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        embeddings = self.module.get_latent_embeddings(*batch)
        self.save_predictions(embeddings, batch_idx)
        
    def get_optimizers(self, zero_all: bool = False) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]]:
        """
        Retrieve optimizers for the model components.

        This function resets gradients if specified and returns a structured dictionary of optimizers.

        Args:
            zero_all (bool, optional): Flag to reset gradients of all optimizers. Defaults to False.

        Returns:
            dict: Dictionary containing optimizers for experts, VAE, and adversarials.
        """
        assert hasattr(self, 'optimizer_map'), "configure_optimizers must be called before get_optimizers"
        optimizers = self.optimizers()

        if zero_all:
            for optim in optimizers:
                optim.zero_grad()

        def replace_indices_with_optimizers(mapping, optimizer_list):
            if isinstance(mapping, dict):
                return {key: replace_indices_with_optimizers(value, optimizer_list) for key, value in mapping.items()}
            else:
                return optimizer_list[mapping]
            
        # Create a dictionary with indices replaced with optimizer instances
        optimizer_dict = replace_indices_with_optimizers(self.optimizer_map, optimizers)
        
        return optimizer_dict
        
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Configure optimizers for different components of the model.

        Returns:
            list: List of configured optimizers for experts, VAE, and adversarials.
        """
        optimizers = {}
        optimizers['experts'] = {
            expert_id: torch.optim.Adam(self.module.experts[expert_id].parameters(), lr=1e-3, weight_decay=1e-6)
            for expert_id in self.module.experts
        }
        optimizers['vae'] = torch.optim.Adam(self.module.vae.encoder.parameters(), lr=1e-3, weight_decay=1e-6)
        optimizers['adversarials'] = {
            i: torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for i, module in enumerate(self.module.adversarials)
        }

        optimizer_list = []
        self.optimizer_map = convert_to_flat_list_and_map(optimizers, optimizer_list)
        return optimizer_list

def convert_to_flat_list_and_map(d: Dict, flat_list: Optional[List] = None) -> Dict:
    """
    Convert all values in the dictionary to a flat list and return the list and a mapping dictionary.
    Args:
        d (dict): The dictionary to convert.
        flat_list (list, optional): The list to append values to. Defaults to None.

    Returns:
        dict: Mapping dictionary linking keys to indices in the flat list.
    """
    if flat_list is None:
        flat_list = []

    map_dict = {}

    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            map_dict[key] = convert_to_flat_list_and_map(value, flat_list)
        else:
            # Add value to flat list and set its index in the mapping
            flat_list.append(value)
            map_dict[key] = len(flat_list) - 1

    return map_dict