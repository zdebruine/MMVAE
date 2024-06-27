from collections import namedtuple
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from sciml.utils.constants import REGISTRY_KEYS as RK

from .mixins.init import HeWeightInitMixIn
from ._vae import VAE
from .base import Experts, BaseModule

class MMVAE(HeWeightInitMixIn, BaseModule):
    """
    Multi-Modal Variational Autoencoder (MMVAE) class.

    This class extends HeWeightInitMixIn and BaseModule, incorporating a VAE and multiple experts for encoding and decoding.
    
    Args:
        vae (VAE): The variational autoencoder module.
        experts (Experts): The experts module containing human and mouse encoders/decoders.
    """
    
    def __init__(
        self,
        vae: VAE,
        experts: Experts
    ):
        super().__init__()
        self.vae = vae
        self.experts = experts
        
        self.init_weights()  # Initialize weights using He initialization
    
    def get_module_inputs(self, batch, **kwargs):
        """
        Prepare the inputs for the module.

        Args:
            batch (dict): A batch of data.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Arguments and keyword arguments for the forward pass.
        """
        args = batch[RK.X], batch[RK.EXPERT_ID]
        return args, kwargs
    
    def forward(self, x, source: str, target: str = None):
        """
        Forward pass through the MMVAE.

        Args:
            x (torch.Tensor): Input tensor.
            source (str): Source expert ID.
            target (str, optional): Target expert ID. Defaults to None.

        Returns:
            tuple: The latent variables and reconstructed outputs.
        """
        if target:
            experts = {target: self.experts[target]}
        else:
            experts = self.experts
            
        shared_x = self.experts[source].encode(x)  # Encode input using the source expert
        qz, pz, shared_x_hat = self.vae(shared_x)  # Pass through the VAE
        
        expert_x_hats = {}
        for expert_id, expert in experts.items():
            x_hat = expert.decode(shared_x_hat)  # Decode the shared representation using each expert
            expert_x_hats[expert_id] = x_hat
            
        if target:
            return qz, pz, expert_x_hats[target]
        
        return qz, pz, expert_x_hats
    
    def configure_optimizers(self):
        """
        Configure optimizers for the VAE and expert modules.

        Returns:
            tuple: Optimizers for the VAE and experts.
        """
        return (
            torch.optim.Adam(self.vae.parameters()),
            torch.optim.Adam(self.experts[RK.HUMAN].parameters()),
            torch.optim.Adam(self.experts[RK.MOUSE].parameters())
        )
    
    def cross_generate(self, x, source):
        """
        Perform cross-generation between species.

        Args:
            x (torch.Tensor): Input tensor.
            source (str): Source expert ID.

        Returns:
            torch.Tensor: Reconstructed tensor from the source after cross-generation.
        """
        target = RK.MOUSE if source == RK.HUMAN else RK.HUMAN
        
        _, _, target_x_hat = self(x, source, target=target)
        _, _, source_cross_x_hat = self(target_x_hat, target, target=source)
        
        return source_cross_x_hat
    
    def loss(
        self, 
        model_inputs, 
        model_outputs,
        kl_weight: float = 1.0,
        use_cross_gen_loss: bool = False,
    ):
        """
        Compute the loss for the MMVAE.

        Args:
            model_inputs (tuple): Inputs to the model.
            model_outputs (tuple): Outputs from the model.
            kl_weight (float, optional): Weight for the KL divergence term. Defaults to 1.0.
            use_cross_gen_loss (bool, optional): Whether to use cross-generation loss. Defaults to False.

        Returns:
            dict: Dictionary containing loss components.
        """
        args, kwargs = model_inputs
        
        x, expert_id = args
        qz, pz, expert_x_hats = model_outputs

        # Compute ELBO (Evidence Lower Bound) loss
        z_kl_div, recon_loss, loss = self.vae.elbo(qz, pz, x, expert_x_hats[expert_id], kl_weight=kl_weight)
        
        cross_gen_loss = {}
        if use_cross_gen_loss:
            if self.training:
                raise RuntimeError("Cannot compute cross gen loss in training mode")
            
            cross_expert_x_hat = self.cross_generate(x, expert_id)
            cross_loss = F.mse_loss(cross_expert_x_hat, x, reduction='mean')
            cross_gen_loss[f"cross_gen_loss/{expert_id}"] = cross_loss
        
        return {
            **cross_gen_loss,
            RK.RECON_LOSS: recon_loss,
            RK.KL_LOSS: z_kl_div,
            RK.LOSS: loss,
            RK.KL_WEIGHT: kl_weight
        }
