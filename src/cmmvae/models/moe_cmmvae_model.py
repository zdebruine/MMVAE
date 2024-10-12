from typing import Optional, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn

from cmmvae.models import BaseModel
from cmmvae.modules import MOE_CMMVAE
from cmmvae.constants import REGISTRY_KEYS as RK


class MOE_CMMVAEModel(BaseModel):
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

    def __init__(self, module: MOE_CMMVAE, adv_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self.automatic_optimization = (
            False  # Disable automatic optimization for manual control
        )
        # Criterion for adversarial loss
        self.adversarial_criterion = nn.BCELoss(reduction="sum")
        self.init_weights()
        self.adv_weight = adv_weight

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

        human_label = torch.zeros(self.batch_size, 1, device=self.device)
        mouse_label = torch.ones(self.batch_size, 1, device=self.device)

        expert_labels = human_label if expert_id == RK.HUMAN else mouse_label
        trick_labels = human_label if expert_id == RK.MOUSE else mouse_label

        # Retrieve optimizers.
        optims = self.get_optimizers(zero_all=True)
        expert_optimizer = optims["experts"][expert_id]
        human_optimizer = optims[RK.HUMAN]
        mouse_optimizer = optims[RK.MOUSE]
        shared_optimizer = optims[RK.SHARED]
        adversarial_optimizers = optims["adversarials"]
        human_gan_optimizer = optims["ganh"]
        mouse_gan_optimizer = optims["ganm"]

        # Perform forward pass and compute the loss
        qz_dict, pz_dict, z_dict, xhats, hidden_representations = self.module(
            x, metadata, expert_id
        )

        loss_dict = {}

        # Calculate loss for cisgenerated specific VAE and shared. Only need KL-Divergence to be added to shared loss.
        # Pytorch backprop will send reconstruction to both and correct kl to each.
        if expert_id == RK.HUMAN:
            # FIXME check if this wrong. Modules or .vaes?
            loss_dict[RK.HUMAN_LOSS] = self.module.vaes[RK.HUMAN].elbo(
                qz_dict[RK.HUMAN],
                pz_dict[RK.HUMAN],
                x,
                xhats["cis"],
                self.kl_annealing_fn.kl_weight,
            )

            loss_dict[RK.SHARED] = self.module.vaes[RK.SHARED].elbo(
                qz_dict[RK.SHARED],
                pz_dict[RK.SHARED],
                x,
                xhats["cis"],
                self.kl_annealing_fn.kl_weight,
            )

            loss_dict[RK.SHARED][RK.LOSS] += (
                loss_dict[RK.HUMAN_LOSS][RK.KL_WEIGHT]
                * loss_dict[RK.HUMAN_LOSS][RK.KL_LOSS]
            )

        else:
            loss_dict[RK.MOUSE_LOSS] = self.module.vaes[RK.MOUSE].elbo(
                qz_dict[RK.MOUSE],
                pz_dict[RK.MOUSE],
                x,
                xhats["cis"],
                self.kl_annealing_fn.kl_weight,
            )

            loss_dict[RK.SHARED] = self.module.vaes[RK.SHARED].elbo(
                qz_dict[RK.SHARED],
                pz_dict[RK.SHARED],
                x,
                xhats["cis"],
                self.kl_annealing_fn.kl_weight,
            )

            loss_dict[RK.SHARED][RK.LOSS] += (
                loss_dict[RK.MOUSE_LOSS][RK.KL_WEIGHT]
                * loss_dict[RK.MOUSE_LOSS][RK.KL_LOSS]
            )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Train adversarial networks
        adversarial_loss = None
        for i, (hidden_rep, adv) in enumerate(
            zip(hidden_representations, self.module.adversarials)
        ):
            # Get adversarial predictions
            adv_output = adv(hidden_rep)

            # Calculate adversarial loss
            current_discriminator_loss = self.adversarial_criterion(
                adv_output, expert_labels
            )

            loss_dict[f"adv_{i}"] = current_discriminator_loss

            # Backpropagation for the adversarial
            self.manual_backward(current_discriminator_loss, retain_graph=True)
            adversarial_optimizers[i].step()

        for i, (hidden_rep, adv) in enumerate(
            zip(hidden_representations, self.module.adversarials)
        ):
            # Get adversarial predictions
            adv_output = adv(hidden_rep)

            # Calculate adversarial loss
            current_adversarial_loss = self.adversarial_criterion(
                adv_output, trick_labels
            )
            # print(current_adversarial_loss)
            if adversarial_loss is None:
                adversarial_loss = current_adversarial_loss
            else:
                adversarial_loss += current_adversarial_loss

        loss_dict["adversarial_loss"] = adversarial_loss

        # FIXME right now only does loss for cross generation. IE human cell -> mouse gan loss.
        # Can also add cis gan loss to vae backprop. IE human loss = recon + adversarial + gan could be loss.
        # Calculate GAN loss
        if expert_id == RK.HUMAN:
            gan_loss = None
            for i, (hidden_rep, ganm) in enumerate(
                zip(hidden_representations, self.module.ganm)
            ):
                # Get gan predictions
                gan_output = ganm(xhats["cross"])

                # Calculate gan loss
                current_gan_loss = self.adversarial_criterion(gan_output, expert_labels)

                loss_dict[f"gan_{i}"] = current_gan_loss

                # Backpropagation for the gans
                self.manual_backward(current_gan_loss, retain_graph=True)
                mouse_gan_optimizer[i].step()

            for i, (hidden_rep, ganm) in enumerate(
                zip(hidden_representations, self.module.ganm)
            ):
                # Get adversarial predictions
                gan_output = ganm(xhats["cross"])

                # Calculate adversarial loss
                current_gan_loss = self.adversarial_criterion(gan_output, trick_labels)
                # print(current_adversarial_loss)
                if gan_loss is None:
                    gan_loss = current_gan_loss
                else:
                    gan_loss += current_gan_loss

            # cis_gan_loss = None
            # for i, (hidden_rep, ganh) in enumerate(
            #     zip(hidden_representations, self.module.ganh)
            # ):
            #     # Get gan predictions
            #     cis_gan_output = ganh(xhats["cis"])

            #     # Calculate gan loss
            #     current_cis_gan_loss = self.adversarial_criterion(
            #         cis_gan_output, expert_labels
            #     )

            #     loss_dict[f"cis_gan_{i}"] = current_cis_gan_loss

            #     # Backpropagation for the gans
            #     self.manual_backward(current_cis_gan_loss, retain_graph=True)
            #     human_gan_optimizer[i].step()

            # for i, (hidden_rep, ganh) in enumerate(
            #     zip(hidden_representations, self.module.ganh)
            # ):
            #     # Get adversarial predictions
            #     cis_gan_output = ganh(xhats["cis"])

            #     # Calculate adversarial loss
            #     current_cis_gan_loss = self.adversarial_criterion(
            #         cis_gan_output, trick_labels
            #     )
            #     # print(current_adversarial_loss)
            #     if cis_gan_loss is None:
            #         cis_gan_loss = current_cis_gan_loss
            #     else:
            #         cis_gan_loss += current_cis_gan_loss
        else:
            gan_loss = None
            for i, (hidden_rep, ganh) in enumerate(
                zip(hidden_representations, self.module.ganh)
            ):
                # Get gan predictions
                gan_output = ganh(xhats["cross"])

                # Calculate gan loss
                current_gan_loss = self.adversarial_criterion(gan_output, expert_labels)

                loss_dict[f"gan_{i}"] = current_gan_loss

                # Backpropagation for the gans
                self.manual_backward(current_gan_loss, retain_graph=True)
                human_gan_optimizer[i].step()

            for i, (hidden_rep, ganh) in enumerate(
                zip(hidden_representations, self.module.ganh)
            ):
                # Get adversarial predictions
                gan_output = ganh(xhats["cross"])

                # Calculate adversarial loss
                current_gan_loss = self.adversarial_criterion(gan_output, trick_labels)
                # print(current_adversarial_loss)
                if gan_loss is None:
                    gan_loss = current_gan_loss
                else:
                    gan_loss += current_gan_loss

            # cis_gan_loss = None
            # for i, (hidden_rep, ganm) in enumerate(
            #     zip(hidden_representations, self.module.ganm)
            # ):
            #     # Get gan predictions
            #     cis_gan_output = ganm(xhats["cis"])

            #     # Calculate gan loss
            #     current_cis_gan_loss = self.adversarial_criterion(
            #         cis_gan_output, expert_labels
            #     )

            #     loss_dict[f"cis_gan_{i}"] = current_cis_gan_loss

            #     # Backpropagation for the gans
            #     self.manual_backward(current_cis_gan_loss, retain_graph=True)
            #     mouse_gan_optimizer[i].step()

            # for i, (hidden_rep, ganm) in enumerate(
            #     zip(hidden_representations, self.module.ganm)
            # ):
            #     # Get adversarial predictions
            #     cis_gan_output = ganm(xhats["cis"])

            #     # Calculate adversarial loss
            #     current_cis_gan_loss = self.adversarial_criterion(
            #         cis_gan_output, trick_labels
            #     )
            #     # print(current_adversarial_loss)
            #     if cis_gan_loss is None:
            #         cis_gan_loss = current_cis_gan_loss
            #     else:
            #         cis_gan_loss += current_cis_gan_loss

        loss_dict["gan_loss"] = gan_loss
        # loss_dict["cis_gan_loss"] = cis_gan_loss

        # If cis-generated, backprop reconstruction loss and kl to specific vae. Otherwise Adversarial loss
        if expert_id == RK.HUMAN:
            self.freeze_model_parameters(self.module)
            self.unfreeze_model_parameters(self.module.vaes[RK.MOUSE])
            self.manual_backward(gan_loss, retain_graph=True)
            self.unfreeze_model_parameters(self.module)
        else:
            self.freeze_model_parameters(self.module)
            self.unfreeze_model_parameters(self.module.vaes[RK.HUMAN])
            self.manual_backward(gan_loss, retain_graph=True)
            self.unfreeze_model_parameters(self.module)

        # Can also add GAN loss. Test which is better.
        s_loss = loss_dict[RK.SHARED][RK.LOSS] + (self.adv_weight * adversarial_loss)
        # + cis_gan_loss

        # Backpropagation for encoder and decoder. Pytorch should backprop reconstruction to both, and correct kl to each. -Tony
        self.manual_backward(s_loss)

        # Clip gradients for stability
        self.clip_gradients(
            human_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )
        self.clip_gradients(
            mouse_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )
        self.clip_gradients(
            shared_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )
        self.clip_gradients(
            expert_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )

        # Handle inactive conditional modules
        inactive_modules = []
        if self.module.vaes[RK.SHARED].conditionals:
            for cl in self.module.vaes[RK.SHARED].conditionals.layers:
                inactive = cl.unique_conditions - cl.active_condition_modules
                inactive_modules.extend(
                    [cl.conditions[key] for key in cl.conditions if key in inactive]
                )

        for module in inactive_modules:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        # Update the weights
        human_optimizer.step()
        mouse_optimizer.step()
        shared_optimizer.step()
        expert_optimizer.step()
        self.kl_annealing_fn.step()

        # Log the loss. FIXME dict with logging
        self.auto_log(loss_dict[RK.SHARED], tags=[self.stage_name, expert_id])

    def validation_step(self, batch: Tuple[torch.Tensor, pd.DataFrame, str]) -> None:
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id = batch

        # Perform forward pass and compute the loss. Only need shared for validation
        qz_dict, pz_dict, z_dict, xhats, hidden_representations = self.module(
            x, metadata, expert_id
        )

        loss_dict = {}

        loss_dict[RK.SHARED] = self.module.vaes[RK.SHARED].elbo(
            qz_dict[RK.SHARED],
            pz_dict[RK.SHARED],
            x,
            xhats["cis"],
            self.kl_annealing_fn.kl_weight,
        )

        self.auto_log(loss_dict[RK.SHARED], tags=[self.stage_name, expert_id])

    # Alias for validation_step method to reuse for testing
    test_step = validation_step

    def predict_step(
        self, batch: Tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ) -> None:
        """
        Perform a prediction step.

        This step extracts latent embeddings and saves them for analysis.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        embeddings = self.module.get_latent_embeddings(*batch)
        self.save_predictions(embeddings, batch_idx)

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
            for optim in optimizers:
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

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Configure optimizers for different components of the model.

        Returns:
            list: List of configured optimizers for experts, VAE, and adversarials.
        """
        optimizers = {}
        optimizers["experts"] = {
            expert_id: torch.optim.Adam(
                self.module.experts[expert_id].parameters(), lr=1e-3, weight_decay=1e-6
            )
            for expert_id in self.module.experts
        }
        optimizers["shared"] = torch.optim.Adam(
            self.module.vaes.parameters(), lr=1e-3, weight_decay=1e-6
        )
        optimizers["human"] = torch.optim.Adam(
            self.module.vaes.parameters(), lr=1e-3, weight_decay=1e-6
        )
        optimizers["mouse"] = torch.optim.Adam(
            self.module.vaes.parameters(), lr=1e-3, weight_decay=1e-6
        )
        optimizers["adversarials"] = {
            i: torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for i, module in enumerate(self.module.adversarials)
        }
        optimizers["ganh"] = {
            i: torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for i, module in enumerate(self.module.ganh)
        }
        optimizers["ganm"] = {
            i: torch.optim.Adam(module.parameters(), lr=1e-3, weight_decay=1e-6)
            for i, module in enumerate(self.module.ganm)
        }

        optimizer_list = []
        self.optimizer_map = convert_to_flat_list_and_map(optimizers, optimizer_list)
        return optimizer_list

    # Helper function to freeze model parameters
    def freeze_model_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    # Helper function to unfreeze model parameters
    def unfreeze_model_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True


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
