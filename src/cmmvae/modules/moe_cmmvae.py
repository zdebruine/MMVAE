from typing import Optional, Union

import pandas as pd
import torch
from torch import nn

from cmmvae.modules.base import Experts, FCBlock, FCBlockConfig, ExpertGANs
from cmmvae.modules import ExpertVAEs
from cmmvae.constants import REGISTRY_KEYS as RK

from collections import defaultdict


ADVERSERIAL_TYP = Union[Optional[FCBlockConfig], list[Optional[FCBlockConfig]]]


class MOE_CMMVAE(nn.Module):
    """
    Mixture of Experts Conditional Multi-Modal Variational Autoencoder class.

    This class extends the CMMVAE to incorporate human and mouse
    specific VAEs and GANs for a mixture of experts approach
    to enhance learning across multiple modalities.

    Attributes:
        vaes (ExpertVAEs): The conditional latent VAEs used for encoding and decoding.
        adversarials (nn.ModuleList): List of adversarial networks
            applied to the latent space.
        ganh (nn.ModuleList): Human GAN applied to reconstructed inputs.
        ganm (nn.ModuleList): Mouse GAN applied to reconstructed inputs.

    Args:
        vaes (cmmvae.modules.moe_clvae.MOE_CLVAE):
            Human, shared, and mouse VAEs.
        experts (cmmvae.modules.base.Experts):
            Collection of expert networks for different modalities.
        adversarials (
            Union[Optional[cmmvae.modules.base.FCBlockConfig],
            List[Optional[cmmvae.modules.base.FCBlockConfig]]]
        ): Configuration(s) for adversarial networks.
        ganh (
            Union[Optional[cmmvae.modules.base.FCBlockConfig],
            List[Optional[cmmvae.modules.base.FCBlockConfig]]]
        ): Configuration for human GAN.
        ganm (
            Union[Optional[cmmvae.modules.base.FCBlockConfig],
            List[Optional[cmmvae.modules.base.FCBlockConfig]]]
        ): Configuration for mouse GAN.
    """

    def __init__(
        self,
        expert_vaes: ExpertVAEs,
        experts: Experts,
        expert_gans: ExpertGANs = None,
        adversarials: ADVERSERIAL_TYP = None,
    ):
        super().__init__()
        self.vaes = expert_vaes
        self.experts = experts
        self.gans = expert_gans
        self.adversarials = nn.ModuleList(
            [FCBlock(config) for config in adversarials]
        ) if adversarials else None

        self.layer_norms = nn.ModuleDict(
            {
                f"{RK.HIDDEN}_{RK.HUMAN}": 
                nn.LayerNorm(self.vaes[RK.HUMAN].decoder.output_dim),
                f"{RK.HIDDEN}_{RK.SHARED}":
                nn.LayerNorm(self.vaes[RK.SHARED].decoder.output_dim),
                f"{RK.HIDDEN}_{RK.MOUSE}":
                nn.LayerNorm(self.vaes[RK.MOUSE].decoder.output_dim),
                f"{RK.HUMAN}":
                nn.LayerNorm(self.experts[RK.HUMAN].decoder.input_dim),
                f"{RK.MOUSE}":
                nn.LayerNorm(self.experts[RK.MOUSE].decoder.input_dim),
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        metadata: pd.DataFrame,
        expert_id: str,
        cross_generate: bool = False,
    ):
        """
        Forward pass through the MOE_CMMVAE.

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
        # Encode the input using the specified expert network
        shared_x = self.experts[expert_id].encode(x)

        outputs = defaultdict(dict)

        for vae in self.vaes:
            qz, pz, z, x_hat, hidden_reps = self.vaes[vae](shared_x, metadata, species=expert_id)
            outputs[RK.QZ][vae] = qz
            outputs[RK.PZ][vae] = pz
            outputs[RK.Z][vae] = z
            outputs[RK.X_HAT][vae] = x_hat
            outputs[RK.HIDDEN][vae] = hidden_reps

        # Concat the expert and shared VAE outputs
        xhats = {}
        # Decode using all experts.
        for expert in self.experts:
                decoder_input = torch.add(
                    self.layer_norms[f"{RK.HIDDEN}_{expert}"](outputs[RK.X_HAT][expert]),
                    self.layer_norms[f"{RK.HIDDEN}_{RK.SHARED}"](outputs[RK.X_HAT][RK.SHARED])
                )

                decoder_input = self.layer_norms[expert](decoder_input)

                xhats[expert] = self.experts[expert].decode(decoder_input)

        outputs[RK.X_HAT] = xhats

        return outputs[RK.QZ], outputs[RK.PZ], outputs[RK.Z], outputs[RK.X_HAT], outputs[RK.HIDDEN]

    @torch.no_grad()
    def get_latent_embeddings(
        self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str
    ) -> dict[str, torch.Tensor]:
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
        embeddings = {}

        # Tag the metadata with the expert_id
        metadata["species"] = expert_id

        # Encode the input using the specified expert network
        x = self.experts[expert_id].encode(x)
        for vae in self.vaes:
            # Encode using the VAEs
            _, z, _ = self.vaes[vae].encode(x)
            embeddings[f"{vae}_{RK.Z}"] = (z, metadata)

        return embeddings
