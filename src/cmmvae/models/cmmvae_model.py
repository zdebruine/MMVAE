from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer  # type: ignore

from cmmvae.models import BaseModel
from cmmvae.modules import CMMVAE
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.modules.base.components import GradientReversalFunction, Adversarial
from cmmvae.config import AutogradConfig


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

    def __init__(
        self,
        module: CMMVAE,
        adv_weight: Optional[float] = None,
        autograd_config: Optional[AutogradConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module = module
        self.automatic_optimization = (
            False  # Disable automatic optimization for manual control
        )

        self.adversarial_criterion = nn.CrossEntropyLoss(reduction="sum")
        self.init_weights()
        self.adv_weight = adv_weight if adv_weight else 1.0
        self.autograd_config = autograd_config or AutogradConfig()

    def grf(
        self,
        hidden_representations: list[torch.Tensor],
        labels: list[torch.Tensor],
        expert_id: str,
        detach: bool = False,
    ):
        adv_losses = []
        for i, (hidden_rep, adversary) in enumerate(
            zip(hidden_representations, self.module.adversarials),
            start = 1
        ):
            head_losses = []
            if detach:
                hidden_rep = hidden_rep.detach()
                loss_tag = f"discriminator_{i}"
            else:
                # Apply Gradient Reversal Function when updating the main network
                hidden_rep = GradientReversalFunction.apply(hidden_rep, 1)
                loss_tag = f"generator_{i}"

            encoded = adversary.encoder(hidden_rep)

            # Calculate adversarial loss
            for condition, label in labels.items():
                predictions = adversary.heads[condition](encoded)
                disc_loss = self.adversarial_criterion(predictions, label)
                head_losses.append(disc_loss)
                self.auto_log(
                    {condition: disc_loss},
                    tags=[loss_tag, self.stage_name, expert_id, RK.ADV_LOSS],
                    key_pos="last",
                )

            summed = torch.sum(torch.stack(head_losses))
            self.auto_log(
                {"summed": summed},
                tags=[loss_tag, self.stage_name, expert_id, RK.ADV_LOSS],
                key_pos="last",
            )
            adv_losses.append(summed)

        return adv_losses

    def gradient_reversal_domain_classifier(
        self,
        hidden_representations: list[torch.Tensor],
        metadata: pd.DataFrame,
        expert_id: str,
        adversarial_optimizers: dict[int, Optimizer],
    ):
        assert self.module.adversarials
        labels = {}
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for condition, map in Adversarial.labels.items():
            vals = metadata[condition].values
            labels[condition] = torch.tensor([map[v] for v in vals], device=device)

        # Compute adversarial loss for adversarial networks
        adv_losses = self.grf(hidden_representations, labels, expert_id, detach=True)
        # Backpropagate adversarial loss to adversarial networks
        for i, (adv_loss, adv_optimizer) in enumerate(zip(adv_losses, adversarial_optimizers.values()), start=1):
            self.manual_backward(adv_loss)
            self.log_gradient_norms(
                {f"discriminator_{i}": adv_optimizer}, tag_prefix="grad_norms"
            )
            # Clip and update adversarial networks
            if self.autograd_config.adversarial_gradient_clip:
                self.clip_gradients(
                    adv_optimizer, *self.autograd_config.adversarial_gradient_clip
                )
            adv_optimizer.step()
            adv_optimizer.zero_grad()

        # Now compute adversarial loss for main network (with gradient reversal)
        adv_losses_main = self.grf(hidden_representations, labels, expert_id, detach=False)
        # Add adversarial loss to total loss (with weight)
        return adv_losses_main

    def training_step(
        self, batch: tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ) -> None:
        x, metadata, expert_id = batch
        metadata["species"] = expert_id

        # Get optimizers
        optims = self.get_optimizers()
        expert_optimizer = optims["experts"][expert_id]
        vae_optimizer = optims["vae"]
        adversarial_optimizers = optims.get("adversarials")

        # Zero all gradients
        vae_optimizer.zero_grad()
        expert_optimizer.zero_grad()
        if adversarial_optimizers:
            for optim in adversarial_optimizers.values():
                optim.zero_grad()

        # Perform forward pass
        qz, pz, z, xhats, hidden_representations = self.module(
            x=x, metadata=metadata, expert_id=expert_id
        )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        main_loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        main_loss_dict["Mean"] = qz.mean.mean()
        main_loss_dict["Variance"] = qz.variance.mean()

        total_loss = main_loss_dict[RK.LOSS]

        adv_losses = None
        # Train adversarial networks
        if self.module.adversarials:
            adv_losses = self.gradient_reversal_domain_classifier(
                hidden_representations, metadata, expert_id, adversarial_optimizers
            )

        if adv_losses:
            for adv_loss in adv_losses:
                total_loss = total_loss + adv_loss * self.adv_weight

        # Backpropagate main loss
        self.manual_backward(total_loss)

        main_loss_dict[RK.LOSS] = total_loss

        self.log_gradient_norms(
            {"vae": vae_optimizer, f"expert_{expert_id}": expert_optimizer},
            tag_prefix="grad_norms",
        )

        if adversarial_optimizers:
            for key, optim in adversarial_optimizers.items():
                self.log_gradient_norms(
                    {f"generator_{key}": optim}, tag_prefix="grad_norms"
                )

        # Clip gradients for stability
        if self.autograd_config.vae_gradient_clip:
            self.clip_gradients(vae_optimizer, *self.autograd_config.vae_gradient_clip)

        if self.autograd_config.expert_gradient_clip:
            self.clip_gradients(
                expert_optimizer, *self.autograd_config.expert_gradient_clip
            )

        # Update the weights
        vae_optimizer.step()
        expert_optimizer.step()
        self.kl_annealing_fn.step()

        # Log the loss
        self.auto_log(main_loss_dict, tags=[self.stage_name, expert_id])

    def validation_step(self, batch: tuple[torch.Tensor, pd.DataFrame, str]):
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
        self, batch: tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
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
        optim_dict = {}
        optim_dict["experts"] = {
            expert_id: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
            for expert_id, module in self.module.experts.items()
        }
        optim_dict["vae"] = optim_cls(
            self.module.vae.parameters(), lr=5e-3, weight_decay=1e-6
        )
        if self.module.adversarials:
            optim_dict["adversarials"] = {
                i: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
                for i, module in enumerate(self.module.adversarials, start=1)
            }

        optimizers = []
        self.optimizer_map = convert_to_flat_list_and_map(optim_dict, optimizers)

        return optimizers


def convert_to_flat_list_and_map(d: dict, flat_list: Optional[list] = None) -> dict:
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
