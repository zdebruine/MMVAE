import random
from typing import Optional

import anndata as ad
import scib
from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation
from sklearn.metrics import silhouette_score
from torchmetrics.functional import pairwise_euclidean_distance

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
        use_cycle_consistency: bool = False,
        measure_integration: bool = False,
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
        self.use_cycle_consistency = use_cycle_consistency
        self.measure_integration = measure_integration

        if measure_integration:
            self.latents = []
            self.latent_md = []
            # self.mus = []
            # self.vars = []

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
    
    # def cycle_consistency(
    #         self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str
    # ) -> torch.Tensor:
    #     perturbed_metadata = metadata.copy(deep=True)
    #     for condition in self.module.vae.conditionals.layers.keys():
    #         if condition == "species":
    #             if random.choice([True, False]):
    #                 new_species = RK.HUMAN if expert_id == RK.MOUSE else RK.MOUSE
    #                 perturbed_metadata["species"] = new_species
    #             else:
    #                 new_species = expert_id
    #         else:
    #             perturbations = random.choices(
    #                 list(
    #                     self.module.vae.conditionals.layers[condition].conditions.keys()
    #                 ),
    #                 k=len(perturbed_metadata)
    #             )
    #             perturbed_metadata[condition] = perturbations

    #     qz, pz, z, xhats, hidden_representations = self.module(
    #         x=x, metadata=perturbed_metadata, encoder_expert_id=expert_id, decoder_expert_id=new_species
    #     )

    #     kl1 = self.module.vae.kl_loss(qz, pz)

    #     qz, pz, z, xhats, hidden_representations = self.module(
    #         x=xhats[new_species], metadata=metadata, encoder_expert_id=new_species, decoder_expert_id=expert_id
    #     )

    #     return qz, pz, z, xhats, hidden_representations, kl1

    def cycle_consistency(
        self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str
    ) -> torch.Tensor:
        perturbed_metadata = metadata.copy(deep=True)

        new_species = RK.HUMAN if expert_id == RK.MOUSE else RK.MOUSE
        perturbed_metadata["species"] = new_species

        qz, pz, z, xhats, hidden_representations = self.module(
            x=x, metadata=perturbed_metadata, encoder_expert_id=expert_id, decoder_expert_id=new_species
        )

        kl1 = self.module.vae.kl_loss(qz, pz)

        qz, pz, z, xhats, hidden_representations = self.module(
            x=xhats[new_species], metadata=metadata, encoder_expert_id=new_species, decoder_expert_id=expert_id
        )

        return qz, pz, z, xhats, hidden_representations, kl1

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
            x=x, metadata=metadata, encoder_expert_id=expert_id
        )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        main_loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        dist_loss_dict = {}

        dist_loss_dict["Mu_Mean"] = qz.mean.mean()
        dist_loss_dict["Mu_STD"] = qz.mean.std()
        dist_loss_dict["Variance_Mean"] = qz.variance.mean()
        dist_loss_dict["Variance_STD"] = qz.variance.std()

        self.auto_log(
            dist_loss_dict,
            tags=["Dist", expert_id],
            key_pos="last",
            on_step=False
        )

        if self.use_cycle_consistency:

            kl1 = self.module.vae.kl_loss(qz, pz)

            # Perform fairness cycle-consistency pass (no perturbations)
            qz, pz, z, xhats, hidden_representations = self.module(
                x=xhats[expert_id], metadata=metadata, encoder_expert_id=expert_id
            )

            # Perform cycle-consistency pass
            cycle_qz, cycle_pz, cycle_z, cycle_xhats, cycle_hidden_representations, cycle_kl1 = self.cycle_consistency(x, metadata, expert_id)

            main_loss_dict[f"{RK.KL_LOSS}2"] = main_loss_dict[RK.KL_LOSS]
            main_loss_dict[RK.KL_LOSS] = kl1

            cycle_loss_dict = self.module.vae.elbo(
                cycle_qz, cycle_pz, x, cycle_xhats[expert_id], self.kl_annealing_fn.kl_weight
            )

            main_loss_dict[f"cycle_{RK.KL_LOSS}"] = cycle_kl1
            main_loss_dict[f"cycle_{RK.KL_LOSS}2"] = cycle_loss_dict[RK.KL_LOSS]
            main_loss_dict[f"cycle_{RK.RECON_LOSS}"] = cycle_loss_dict[RK.RECON_LOSS]

            total_loss = main_loss_dict[RK.LOSS] + cycle_loss_dict[RK.LOSS] + cycle_kl1 * self.kl_annealing_fn.kl_weight + kl1 * self.kl_annealing_fn.kl_weight
        else:
            total_loss = main_loss_dict[RK.LOSS]

        adv_losses = {}
        # Train adversarial networks
        if self.module.adversarials:

            # real_out = self.module.adversarials[expert_id](x)
            # label = torch.ones_like(real_out).to(
            #     real_out.device
            # )

            # adv_losses[RK.X] = self.adversarial_criterion(real_out, label)
            # total_loss = total_loss + adv_losses[RK.X] * self.adv_weight

            # for expert, xhat in xhats.items():
            #     fake_out = self.module.adversarials[expert](xhat, gradient_reversal=True)
            #     fake_label = torch.zeros_like(fake_out).to(
            #         fake_out.device
            #     )
            #     adv_losses[f"{expert_id}_to_{expert}"] = self.adversarial_criterion(
            #         fake_out, fake_label
            #     )
            #     total_loss = total_loss + adv_losses[f"{expert_id}_to_{expert}"] * self.adv_weight

            # idxs = metadata[RK.SPECIES].map(self.module.adversarials[RK.SPECIES].labels).values  # numpy array
            # labels = torch.nn.functional.one_hot(
            #     torch.as_tensor(idxs, device=x.device, dtype=torch.long),
            #     num_classes=len(self.module.adversarials[RK.SPECIES].labels)
            # ).float()

            # cond_out = self.module.adversarials[RK.SPECIES](
            #     hidden_representations[f"cross_{RK.SPECIES}"], detach=True, gradient_reversal=True
            # )
            # adv_losses[f"cross_{RK.SPECIES}"] = self.adversarial_criterion(
            #     cond_out, labels
            # )
            # total_loss = total_loss + adv_losses[f"cross_{RK.SPECIES}"] * self.adv_weight

            for condition in self.module.vae.conditionals.layers.keys():

                idxs = metadata[condition].map(self.module.adversarials[condition].labels).values  # numpy array
                labels = torch.nn.functional.one_hot(
                    torch.as_tensor(idxs, device=x.device, dtype=torch.long),
                    num_classes=len(self.module.adversarials[condition].labels)
                ).float()

                z_out = self.module.adversarials[condition](
                    hidden_representations[RK.Z_STAR], gradient_reversal=True
                )
                adv_losses[condition] = self.adversarial_criterion(
                    z_out, labels
                )
                total_loss = total_loss + adv_losses[condition] * self.adv_weight

                # cond_out = self.module.adversarials[condition](
                #     hidden_representations[condition], detach=True
                # )
                # adv_losses[condition] = self.adversarial_criterion(
                #     cond_out, labels
                # )
                # total_loss = total_loss + adv_losses[condition] * self.adv_weight

            self.auto_log(adv_losses, tags=[RK.ADV_LOSS, expert_id], key_pos="last")

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
                    {key: optim}, tag_prefix="grad_norms"
                )

        # Clip gradients for stability
        if self.autograd_config.vae_gradient_clip:
            self.clip_gradients(vae_optimizer, *self.autograd_config.vae_gradient_clip)

        if self.autograd_config.expert_gradient_clip:
            self.clip_gradients(
                expert_optimizer, *self.autograd_config.expert_gradient_clip
            )

        if adversarial_optimizers:
            for optim in adversarial_optimizers.values():
                if self.autograd_config.adversarial_gradient_clip:
                    self.clip_gradients(
                        optim, *self.autograd_config.adversarial_gradient_clip
                    )
                optim.step()

        # Update the weights
        vae_optimizer.step()
        expert_optimizer.step()
        self.kl_annealing_fn.step()

        # Log the loss
        self.auto_log(main_loss_dict, tags=[self.stage_name, expert_id])

    def on_validation_epoch_start(self):
        # self.X.clear()
        if self.measure_integration:
            self.latents.clear()
            self.latent_md.clear()
            # self.mus.clear()
            # self.vars.clear()
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking or not self.measure_integration:
            return super().on_validation_epoch_end()
        
        # mus = torch.cat(self.mus, dim=0).cpu().numpy()
        # vars = torch.cat(self.vars, dim=0).cpu().numpy()

        # mu_mean = mus.mean(dim=0)
        # mu_std  = mus.std(dim=0)
        # var_mean = vars.mean(dim=0)
        # var_std  = vars.std(dim=0)

        # distribution_stats = {
        #     "mu_mean": mu_mean,
        #     "mu_std": mu_std,
        #     "var_mean": var_mean,
        #     "var_std": var_std
        # }
        
        # integration = {}

        # X = torch.cat(self.X, dim=0).cpu().numpy()
        latents = torch.cat(self.latents, dim=0).cpu().numpy()
        latent_md = pd.concat(self.latent_md, axis=0, ignore_index=True)

        # distances = pairwise_euclidean_distance(latents)
        # distances = distances.cpu().numpy()

        # integration["species"] = silhouette_score(distances, latent_md["species"].values, metric="precomputed")
        # integration["cell_type"] = silhouette_score(distances, latent_md["cell_type"].values, metric="precomputed")

        # self.auto_log(
        #     integration,
        #     tags=["integration"],
        #     key_pos="last",
        # )

        # Wrap embedding & metadata into AnnData for SCIB
        adata = ad.AnnData(X= None, obs= latent_md)
        # adata.obs = latent_md.copy()
        adata.obsm['X_emb'] = latents

        # bm = Benchmarker(
        #     adata,
        #     batch_key='species',
        #     label_key='cell_type',
        #     embedding_obsm_keys=['X_emb'],
        #     batch_correction_metrics=BatchCorrection(
        #         silhouette_batch=True, ilisi_knn=False, kbet_per_label=False, graph_connectivity=False, pcr_comparison=False
        #     ),
        #     bio_conservation_metrics=BioConservation(
        #         silhouette_label=True, nmi_ari_cluster_labels_kmeans=False, isolated_labels=True, clisi_knn=False
        #     ),
        #     n_jobs=4,           # parallel neighbor search
        # )

        # 3) Run (no need to call bm.prepare() since X=None skips graph metrics)
        # bm.prepare()            # builds any necessary graphs (optional here)
        # bm.benchmark()          # runs embedding & any enabled graph metrics

        # 4) Retrieve results as DataFrame
        # results = bm.get_results(min_max_scale=True, clean_names=True)

        results = scib.metrics.metrics(
            adata,
            adata,                    # integrated AnnData (same as input if only embeddings)
            batch_key='species',      # your batch column in adata.obs
            label_key='cell_type',    # your biological label column
            embed='X_emb',            # name of your embedding in adata.obsm
            silhouette_=True,         # compute both batch and label silhouettes
            ari_=True,                # adjusted Rand index
            nmi_=True,                # normalized mutual information
            ilisi_=False,              # integration LISI
            clisi_=False,              # cell‐type LISI
            isolated_labels_asw_=False,
            isolated_labels_f1_=False,
            # disable all graph‐based metrics:
            graph_conn_=False,
            kBET_=False,
            pcr_=False,
            hvg_score_=False,
            cell_cycle_=False
        )

        print(results)
        print(results.columns)
        print(results.index)

        # Convert results DataFrame to a dict: {metric_name: score}
        # integration = dict(zip(results['metric'], results['score']))
        integration = results[0].dropna().to_dict()

        self.auto_log(
            integration,
            tags=["integration"],
            key_pos="last",
        )

        self.latents.clear()
        self.latent_md.clear()

        return super().on_validation_epoch_end()

    def validation_step(self, batch: tuple[torch.Tensor, pd.DataFrame, str]):
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id = batch
        metadata["species"] = expert_id

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, hidden_representations = self.module(x, metadata, expert_id)

        if self.measure_integration:
            # self.X.append(x)
            self.latent_md.append(metadata)
            self.latents.append(hidden_representations[RK.Z_STAR].cpu())

        # if self.use_cycle_consistency:
        #     # Perform fairness cycle-consistency pass (no perturbations)
        #     qz, pz, z, xhats, hidden_representations = self.module(
        #         x=xhats[expert_id], metadata=metadata, encoder_expert_id=expert_id
        #     )

        #     # Perform cycle-consistency pass
        #     cycle_qz, cycle_pz, cycle_z, cycle_xhats, cycle_hidden_representations = self.cycle_consistency(x, metadata, expert_id)

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        main_loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        # if self.use_cycle_consistency:
        #     cycle_loss_dict = self.module.vae.elbo(
        #         cycle_qz, cycle_pz, x, cycle_xhats[expert_id], self.kl_annealing_fn.kl_weight
        #     )

        #     main_loss_dict[f"cycle_{RK.KL_LOSS}"] = cycle_loss_dict[RK.KL_LOSS]
        #     main_loss_dict[f"cycle_{RK.RECON_LOSS}"] = cycle_loss_dict[RK.RECON_LOSS]

        #     main_loss_dict[RK.LOSS] = main_loss_dict[RK.LOSS] + cycle_loss_dict[RK.LOSS]

        self.auto_log(main_loss_dict, tags=[self.stage_name, expert_id])

        if self.trainer.validating:
            self.log("val_loss", main_loss_dict[RK.LOSS], logger=False, on_epoch=True)

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
                key: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
                for key, module in self.module.adversarials.items()
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
