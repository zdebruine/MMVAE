from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer  # type: ignore

from cmmvae.models import CMMVAEModel, convert_to_flat_list_and_map
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.modules import MOE_CMMVAE

class MOE_CMMVAEModel(CMMVAEModel):
    r"""
    Mixture of Experts Conditional Multi-Modal Variational Autoencoder (MOE_CMMVAE) model for handling expert-specific data.

    This class is designed for training VAEs with multiple experts and adversarial components.

    Args:
        module (Any): Mixture of Experts Conditional Multi-Modal VAE module.
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
        module (`MOE_CMMVAE`): The MOE_CMMVAE module for processing and generating data.
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
        adversarial_criterion (nn.CrossEntropyLoss): Loss function for adversarial training.
        kl_annealing_fn (cmmvae.modules.base.KLAnnealingFn): KLAnnealingFn for weighting KL Divergence. Defaults to KLAnnealingFn(1.0).
    """

    def __init__(
        self,
        module: MOE_CMMVAE,
        gan_weight=None,
        gan_method="",
        *args,
        **kwargs,
    ):
        super().__init__(module=module, *args, **kwargs)
        # Criterion for gan loss
        if gan_method == "GRF":
            gan_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            gan_criterion = nn.BCELoss(reduction="sum")
        self.gan_criterion = gan_criterion
        self.gan_weight = gan_weight if gan_weight else 1.0
        self.gan_method = gan_method
    
    def gan_feedback(
        self,
        xhats,
        expert_id,
        gan_optimizers,
    ):
        batch_size = xhats[expert_id].shape[0]
        human_label = torch.zeros(batch_size, 1, device=self.device)
        mouse_label = torch.ones(batch_size, 1, device=self.device)

        expert_labels = human_label if expert_id == "human" else mouse_label
        trick_labels = human_label if expert_id == "mouse" else mouse_label

        gan_loss = []
        gan_in = torch.nn.functional.layer_norm(
            xhats[expert_id], (xhats[expert_id].shape[1],)
        )
        # Get adversarial predictions
        gan_out = self.module.gans[expert_id](gan_in.detach())

        # Calculate adversarial loss
        current_discriminator_loss = self.gan_criterion(
            gan_out, expert_labels
        )

        self.auto_log({"GAN_Up": current_discriminator_loss}, tags=[self.stage_name, expert_id])

        # Backpropagation for the adversarial
        self.manual_backward(current_discriminator_loss * self.gan_weight, retain_graph=True)

        if self.autograd_config.gan_gradient_clip:
            self.clip_gradients(
                gan_optimizers[expert_id],
                *self.autograd_config.gan_gradient_clip,
            )

        gan_optimizers[expert_id].step()

        cross_species = RK.HUMAN if expert_id == RK.MOUSE else RK.MOUSE
        gan_in = torch.nn.functional.layer_norm(
            xhats[cross_species], (xhats[cross_species].shape[1],)
        )
        # Get adversarial predictions
        gan_output = self.module.gans[cross_species](gan_in)

        # Calculate adversarial loss
        gan_loss = self.gan_criterion(
            gan_output, trick_labels
        )

        self.auto_log({"GAN_Adv": gan_loss}, tags=[self.stage_name, cross_species])

        return gan_loss

    def training_step(
        self, batch: tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
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
        qz_dict, pz_dict, z_dict, xhat_dict, hr_dict = self.module(
            x=x, metadata=metadata, expert_id=expert_id
        )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate loss for human/mouse specific VAE and shared vae. Only need KL-Divergence of specific vae to be added to shared loss.
        # Pytorch autograd will backprop reconstruction to both and the kl that corresponds to each network.
        loss_dict = self.module.vaes[expert_id].elbo(
            qz_dict[expert_id],
            pz_dict[expert_id],
            x,
            xhat_dict[expert_id],
            self.kl_annealing_fn.kl_weight,
        )
        shared_kl = self.module.vaes[RK.SHARED].kl_loss(
            qz_dict[RK.SHARED],
            pz_dict[RK.SHARED],
            self.kl_annealing_fn.kl_weight,
        )
        loss_dict[RK.SHARED_KL] = shared_kl
        total_loss = loss_dict[RK.LOSS]
        total_loss = total_loss + shared_kl

        # Retrieve optimizers.
        cross_species = RK.HUMAN if expert_id == RK.MOUSE else RK.MOUSE
        optims = self.get_optimizers(zero_all=True)
        expert_optimizer = optims[RK.EXPERT][expert_id]
        cross_optimizer = optims[RK.EXPERT][cross_species]
        human_optimizer = optims[RK.VAE][RK.HUMAN]
        mouse_optimizer = optims[RK.VAE][RK.MOUSE]
        shared_optimizer = optims[RK.VAE][RK.SHARED]
        gan_optimizers = optims.get("gans") if self.module.gans else None
        adversarial_optimizers = optims.get("adversarials") if self.module.adversarials else None

        adv_loss = None
        # Train adversarial networks
        if self.module.adversarials:
            if self.adversarial_method == "GRF":
                adv_loss = self.gradient_reversal_domain_classifier(
                    hr_dict[RK.SHARED] + [z_dict[RK.SHARED]], expert_id, adversarial_optimizers
                )
            else:
                adv_loss = self.adversarial_feedback(
                    hr_dict[RK.SHARED], expert_id, adversarial_optimizers
                )

        if adv_loss:
            total_loss = total_loss + adv_loss * self.adv_weight

        gan_loss = None
        # Train gan networks
        if self.module.gans:
            # if self.gan_method == "GRF":
            #     pass
            #     # TODO: GRF method for GAN structure
            #     # gan_loss = self.gradient_reversal_domain_classifier(
            #     #     hr_dict[RK.SHARED] + [z_dict[RK.SHARED]], expert_id, adversarial_optimizers
            #     # )
            # else:
            gan_loss = self.gan_feedback(
                xhat_dict, expert_id, gan_optimizers
            )

        if gan_loss:
            total_loss = total_loss + gan_loss * self.gan_weight

        self.manual_backward(total_loss)

        # Clip gradients for stability
        if self.autograd_config.vae_gradient_clip:
            self.clip_gradients(
                human_optimizer,
                *self.autograd_config.vae_gradient_clip,
            )
            self.clip_gradients(
                mouse_optimizer,
                *self.autograd_config.vae_gradient_clip,
            )
            self.clip_gradients(
                shared_optimizer,
                *self.autograd_config.vae_gradient_clip,
            )
        if self.autograd_config.expert_gradient_clip:
            self.clip_gradients(
                expert_optimizer,
                *self.autograd_config.expert_gradient_clip,
            )
            self.clip_gradients(
                cross_optimizer,
                *self.autograd_config.expert_gradient_clip,
            )

        # Update the weights
        human_optimizer.step()
        mouse_optimizer.step()
        shared_optimizer.step()
        expert_optimizer.step()
        cross_optimizer.step()
        self.kl_annealing_fn.step()

        # Log the loss.
        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])

    def validation_step(self, batch: tuple[torch.Tensor, pd.DataFrame, str]) -> None:
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id = batch

        # Perform forward pass and compute the loss
        qz_dict, pz_dict, z_dict, xhat_dict, hr_dict = self.module(
            x=x, metadata=metadata, expert_id=expert_id
        )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate loss for human/mouse specific VAE and shared vae. Only need KL-Divergence of specific vae to be added to shared loss.
        # Pytorch autograd will backprop reconstruction to both and the kl that corresponds to each network.
        loss_dict = self.module.vaes[expert_id].elbo(
            qz_dict[expert_id],
            pz_dict[expert_id],
            x,
            xhat_dict[expert_id],
            self.kl_annealing_fn.kl_weight,
        )
        shared_kl = self.module.vaes[RK.SHARED].kl_loss(
            qz_dict[RK.SHARED],
            pz_dict[RK.SHARED],
            self.kl_annealing_fn.kl_weight,
        )
        loss_dict[RK.SHARED_KL] = shared_kl
        loss_dict[RK.LOSS] += shared_kl

        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])

        if self.trainer.validating:
            self.log(
                "val_loss", loss_dict[RK.LOSS], logger=True, on_epoch=True
            )

    # Alias for validation_step method to reuse for testing
    test_step = validation_step

    def configure_optimizers(self, optim_cls="Adam") -> list[Optimizer]:  # type: ignore
        """
        Configure optimizers for different components of the model.

        Returns:
            list: List of configured optimizers for experts, VAE, and adversarials.
        """
        optim_cls = Adam if optim_cls == "Adam" else AdamW
        optimizers = {}
        optimizers[RK.EXPERT] = {
            expert_id: optim_cls(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for expert_id, module in self.module.experts.items()
        }
        optimizers[RK.VAE] = {
            expert_id: optim_cls(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for expert_id, module in self.module.vaes.items()
        }
        if self.module.gans:
            optimizers["gans"] = {
                expert_id: optim_cls(module.parameters(), lr=1e-3, weight_decay=1e-6)
                for expert_id, module in self.module.gans.items()
            }
        if self.module.adversarials:
            optimizers["adversarials"] = {
                i: optim_cls(module.parameters(), lr=1e-3, weight_decay=1e-6)
                for i, module in enumerate(self.module.adversarials)
            }
        optimizer_list = []
        self.optimizer_map = convert_to_flat_list_and_map(optimizers, optimizer_list)
        return optimizer_list