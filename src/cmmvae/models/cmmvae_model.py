from typing import Optional, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer  # type: ignore

from cmmvae.models import BaseModel
from cmmvae.modules import CMMVAE
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.modules.base.components import GradientReversalFunction


class CMMVAEModel(BaseModel):
    r"""
    Conditional Multi-Modal Variational Autoencoder (CMMVAE) model for handling expert-specific data.

    This class is designed for training VAEs with multiple experts and adversarial components.

    Args:
        module (Any): Conditional Multi-Modal VAE module.
        batch_size (int, optional): Batch size for logging purposes only. Defaults to 128.
        record_gradients (bool, optional): Whether to record gradients of the model. Defaults to False.
        save_gradients_interval (int): Interval of steps to save gradients. Defaults to 25.
        gradient_record_cap (int, optional): Cap on the number of gradients to record to prevent clogging TensorBoard. Defaults to 20.
        kl_annealing_fn (KLAnnealingFn, optional): Annealing function used for kl_weight. Defaults to `KLAnnealingFn(1.0)`
        predict_dir (str): Directory to save predictions. If not absolute path then saved within Tensorboard log_dir. Defaults to "".
        predict_save_interval (int): Interval to save embeddings and metadata to prevent OOM Error. Defaults to 600.
        initial_save_index (int): The starting point for predictions index when saving (ie z_embeddings_0.npz for -1). Defaults to -1.
        use_he_init_weights (bool): Initialize weights using He initialization. Defaults to True.

    Attributes:
        module (`CMMVAE`): The CMMVAE module for processing and generating data.
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
        adversarial_criterion (nn.CrossEntropyLoss): Loss function for adversarial training.
        kl_annealing_fn (cmmvae.modules.base.KLAnnealingFn): KLAnnealingFn for weighting KL Divergence. Defaults to KLAnnealingFn(1.0).
    """

    def __init__(self, module: CMMVAE, adv_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self.automatic_optimization = (
            False  # Disable automatic optimization for manual control
        )
        # Criterion for adversarial loss
        self.adversarial_criterion = nn.CrossEntropyLoss(reduction="sum")
        self.init_weights()
        self.adv_weight = adv_weight if adv_weight else 1.0

    def shared_adversarial_loss(
        self,
        hidden_representations: list[torch.Tensor],
        expert_id: str,
    ):
        assert self.module.adversarials

        batch_size = hidden_representations[0].shape[0]

        _tensor_fn = torch.zeros if expert_id == "human" else torch.ones
        label = _tensor_fn(batch_size, 1, device=self.device, dtype=torch.float32)

        adv_losses = torch.empty((batch_size, 1), device=self.device)
        for i, (hidden_rep, adv) in enumerate(
            zip(hidden_representations, self.module.adversarials)
        ):
            reverse_hidden_rep = GradientReversalFunction.apply(hidden_rep, 1)
            # Get adversarial predictions
            adv_output = adv(reverse_hidden_rep)

            # Calculate adversarial loss
            disc_loss = self.adversarial_criterion(adv_output, label)
            adv_losses[i] = disc_loss

        return torch.sum(adv_losses) / batch_size

    def training_step(
        self, batch: Tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ) -> None:
        """
        Perform a single training step.

        This involves encoding the input, calculating losses, and updating weights.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        x, metadata, expert_id = batch

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, hidden_representations = self.module(
            x=x, metadata=metadata, expert_id=expert_id
        )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        # Train adversarial networks
        if self.module.adversarials:
            loss_dict[RK.ADV_LOSS] = self.shared_adversarial_loss(
                hidden_representations + [z], expert_id
            )
            loss_dict[RK.LOSS] = (
                loss_dict[RK.LOSS] + loss_dict[RK.ADV_LOSS] * self.adv_weight
            )
            loss_dict[RK.ADV_WEIGHT] = self.adv_weight

        optims = self.get_optimizers(zero_all=True)
        expert_optimizer = optims["experts"][expert_id]
        vae_optimizer = optims["vae"]
        adversarial_optimizers = optims.get("adversarials")

        # Backpropagation for encoder and decoder
        self.manual_backward(loss_dict[RK.LOSS])

        # Clip gradients for stability
        self.clip_gradients(
            vae_optimizer, gradient_clip_val=1, gradient_clip_algorithm="norm"
        )
        self.clip_gradients(
            expert_optimizer, gradient_clip_val=1, gradient_clip_algorithm="norm"
        )

        if adversarial_optimizers:
            for optim in adversarial_optimizers.values():
                self.clip_gradients(
                    optim, gradient_clip_val=1, gradient_clip_algorithm="norm"
                )
                optim.step()

        # Update the weights
        vae_optimizer.step()
        expert_optimizer.step()
        self.kl_annealing_fn.step()

        # Log the loss
        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])

    def validation_step(self, batch: Tuple[torch.Tensor, pd.DataFrame, str]):
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id = batch
        # expert_label = self.module.experts.labels[expert_id]

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, hidden_representations = self.module(x, metadata, expert_id)

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])

        if self.trainer.validating:
            self.log("val_loss", loss_dict[RK.LOSS], logger=False, on_epoch=True)

    # Alias for validation_step method to reuse for testing
    test_step = validation_step

    def predict_step(
        self, batch: Tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ):
        """
        Perform a prediction step.

        This step extracts latent embeddings and saves them for analysis.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        x, metadata, species = batch
        embeddings = self.module.get_latent_embeddings(x, metadata, species)
        return embeddings
        # self.save_predictions(embeddings, batch_idx)

    def get_optimizers(self, zero_all: bool = False):
        """
        Retrieve optimizers for the model components.

        This function resets gradients if specified and returns a structured dictionary of optimizers.

        Args:
            zero_all (bool, optional): Flag to reset gradients of all optimizers. Defaults to False.

        Returns:
            dict: Dictionary containing optimizers for experts, VAE, and adversarials.
        """
        optimizers = self.optimizers()

        if zero_all:
            for optim in optimizers:  # type: ignore
                optim.zero_grad()

        def replace_indices_with_optimizers(mapping, optimizer_list):
            if isinstance(mapping, dict):
                return {
                    key: replace_indices_with_optimizers(value, optimizer_list)
                    for key, value in mapping.items()
                }
            else:
                return optimizer_list[mapping]

        # Create a dictionary with indices replaced with optimizer instances
        optimizer_dict = replace_indices_with_optimizers(self.optimizer_map, optimizers)

        return optimizer_dict

    def configure_optimizers(self, optim_cls="Adam") -> list[Optimizer]:  # type: ignore
        """
        Configure optimizers for different components of the model.

        Returns:
            list: List of configured optimizers for experts, VAE, and adversarials.
        """
        optim_cls = Adam if optim_cls == "Adam" else AdamW
        optimizers = {}
        optimizers["experts"] = {
            expert_id: optim_cls(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for expert_id, module in self.module.experts.items()
        }
        optimizers["vae"] = optim_cls(
            self.module.vae.parameters(), lr=1e-3, weight_decay=1e-6
        )
        if self.module.adversarials:
            optimizers["adversarials"] = {
                i: optim_cls(module.parameters(), lr=1e-3, weight_decay=1e-6)
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
