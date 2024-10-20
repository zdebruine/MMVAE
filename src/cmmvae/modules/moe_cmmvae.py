from typing import Optional, Union

import pandas as pd
import torch
from torch import nn

from cmmvae.modules.base import Experts, FCBlock, FCBlockConfig
from cmmvae.modules import ExpertVAEs
from cmmvae.constants import REGISTRY_KEYS as RK


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
        vaes: ExpertVAEs,
        experts: Experts,
        ganh: ADVERSERIAL_TYP = None,
        ganm: ADVERSERIAL_TYP = None,
        adversarials: ADVERSERIAL_TYP = None,
    ):
        super().__init__()
        self.vaes = vaes
        self.experts = experts
        self.adversarials = nn.ModuleList(
            [FCBlock(config) for config in adversarials if config]
        )
        self.ganh = nn.ModuleList([FCBlock(config) for config in ganh if config])
        self.ganm = nn.ModuleList([FCBlock(config) for config in ganm if config])

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

        # Pass through the VAEs
        (
            qz_human,
            pz_human,
            z_human,
            human_xhat,
            hidden_representations_human,
        ) = self.vaes[RK.HUMAN](shared_x, metadata)
        (
            qz_mouse,
            pz_mouse,
            z_mouse,
            mouse_xhat,
            hidden_representations_mouse,
        ) = self.vaes[RK.MOUSE](shared_x, metadata)
        (
            qz_shared,
            pz_shared,
            z_shared,
            shared_xhat,
            hidden_representations_shared,
        ) = self.vaes[RK.SHARED](shared_x, metadata)

        # Store outputs in dictionaries
        qz_dict = {
            "human": qz_human,
            "mouse": qz_mouse,
            "shared": qz_shared,
        }
        pz_dict = {
            "human": pz_human,
            "mouse": pz_mouse,
            "shared": pz_shared,
        }
        z_dict = {
            "human": z_human,
            "mouse": z_mouse,
            "shared": z_shared,
        }
        hr_dict = {
            "human": hidden_representations_human,
            "mouse": hidden_representations_mouse,
            "shared": hidden_representations_shared,
        }
        xhats = {
            "human": human_xhat,
            "mouse": mouse_xhat,
            "shared": shared_xhat,
            "cis": None,
            "cross": None,
        }

        # define layernorm. Used to learn ratio to sum outputs from all 3 vaes.
        LN = nn.LayerNorm(normalized_shape=human_xhat.shape[-1], device="cuda:0")

        human_xhat = LN(human_xhat)
        mouse_xhat = LN(mouse_xhat)
        shared_xhat = LN(shared_xhat)

        human_xhat = LN(human_xhat)
        mouse_xhat = LN(mouse_xhat)
        shared_xhat = LN(shared_xhat)

        # Sum outputs of VAEs at learned ratio for reconstruction and adversarial backpropagation.
        human_decoder = torch.add(human_xhat, shared_xhat)
        mouse_decoder = torch.add(mouse_xhat, shared_xhat)

        human_decoder = LN(human_decoder)
        mouse_decoder = LN(mouse_decoder)

        # Decode using the specified expert. Label outputs as cis or cross generation.
        if expert_id == RK.HUMAN:
            xhats["cis"] = self.experts[expert_id].decode(human_decoder)
            xhats["cross"] = self.experts[RK.MOUSE].decode(mouse_decoder)
        else:
            xhats["cis"] = self.experts[expert_id].decode(mouse_decoder)
            xhats["cross"] = self.experts[RK.HUMAN].decode(human_decoder)

        return qz_dict, pz_dict, z_dict, xhats, hr_dict

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
        # Encode the input using the specified expert network
        x = self.experts[expert_id].encode(x)

        # Encode using the VAEs
        _, z, _ = self.vaes[RK.SHARED].encode(x)

        # Tag the metadata with the expert_id
        metadata["species"] = expert_id

        return {RK.Z: z, f"{RK.Z}_{RK.METADATA}": metadata}
