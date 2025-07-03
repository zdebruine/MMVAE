from typing import Optional, Union
import warnings

import pandas as pd
import torch
from torch import nn

from cmmvae.modules.base import Experts, FCBlockConfig, Adversarials
from cmmvae.modules import CLVAE
from cmmvae.constants import REGISTRY_KEYS as RK

class CMMVAE(nn.Module):
    """
    Conditional Multi-Modal Variational Autoencoder class.

    This class extends the MMVAE to incorporate conditional
    latent spaces and adversarial networks for
    enhanced learning across multiple modalities.

    Attributes:
        vae (Any): The conditional latent VAE used for encoding and decoding.
        adversarials (nn.ModuleList): List of adversarial networks
            applied to the latent space.

    Args:
        vae (cmmvae.modules.clvae.CLVAE):
            Instance of a conditional latent VAE.
        experts (cmmvae.modules.base.Experts):
            Collection of expert networks for different modalities.
        adversarials (
            Union[Optional[cmmvae.modules.base.FCBlockConfig],
            List[Optional[cmmvae.modules.base.FCBlockConfig]]]
        ): Configuration(s) for adversarial networks.
    """

    def __init__(
        self,
        vae: CLVAE,
        experts: Experts,
        adversarials: Optional[Adversarials] = None,
    ):
        super().__init__()
        self.vae = vae
        self.experts = experts
        self.adversarials = adversarials

    def forward(
        self,
        x: torch.Tensor,
        metadata: pd.DataFrame,
        encoder_expert_id: str,
        decoder_expert_id: str = None,
        cross_generate: bool = False,
    ):
        """
        Forward pass through the CMMVAE.

        This method performs encoding, decoding,
        and optional cross-generation for multi-modal inputs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).
            metadata (pd.DataFrame): Metadata associated with the input data.
            expert_id (str): Identifier for the expert network to use.
            cross_generate (bool, optional):
                Flag to enable cross-generation across experts.
                    Defaults to False.

        Returns:
            tuple:
                - qz (torch.distributions.Distribution):
                    Approximate posterior distribution.
                - pz (torch.distributions.Distribution):
                    Prior distribution.
                - z (torch.Tensor): Sampled latent variable.
                - xhats (Dict[str, torch.Tensor]):
                    Reconstructed outputs for each expert.
                - hidden_representations (List[torch.Tensor]):
                    Hidden representations from the VAE.
        """
        if decoder_expert_id is None:
            decoder_expert_id = encoder_expert_id

        # Encode the input using the specified expert network
        shared_x = self.experts[encoder_expert_id].encode(x)

        # Pass through the VAE
        if cross_generate or decoder_expert_id != encoder_expert_id:
            cross_species = RK.HUMAN if encoder_expert_id == RK.MOUSE else RK.MOUSE
        else:
            cross_species = None

        qz, pz, z, shared_xhats, hidden_representations = self.vae(
            shared_x, metadata, encoder_expert_id, cross_species=cross_species,
        )

        xhats = {}

        # Perform cross-generation if enabled
        if cross_generate:
            # if self.training:
            #     warnings.warn(
            #         """
            #         CMMVAE is cross-generating during training,
            #         which could cause gradients to be
            #         accumulated for cross-generation passes
            #         """
            #     )

            # Decode using all avaialble experts
            for expert in self.experts:
                xhats[expert] = self.experts[expert].decode(shared_xhats[expert])

        else:
            # Decode using the specified expert
            xhats[decoder_expert_id] = self.experts[decoder_expert_id].decode(shared_xhats[decoder_expert_id])

        return qz, pz, z, xhats, hidden_representations

    @torch.no_grad()
    def get_latent_embeddings(
        self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str
    ) -> dict[str, tuple[torch.Tensor, pd.DataFrame]]:
        """
        Obtain latent embeddings from the input data
            using the specified expert network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).
            metadata (pd.DataFrame): Metadata associated with the input data.
            expert_id (str): Identifier for the expert network to use.

        Returns:
            dict: Dictionary containing the following keys and values:
                - RK.Z: Latent embeddings.
                - f"{RK.Z}_{RK.METADATA}": Metadata.
        """
        # Encode the input using the specified expert network
        x = self.experts[expert_id].encode(x)

        # Encode using the VAE
        _, z, _ = self.vae.encode(x)

        # Tag the metadata with the expert_id
        metadata["species"] = expert_id

        return {RK.Z: (z, metadata)}
