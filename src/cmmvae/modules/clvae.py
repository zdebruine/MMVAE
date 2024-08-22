from typing import Optional, Dict, List

import torch
import pandas as pd

from cmmvae.modules.vae import VAE
from cmmvae.modules.base import FCBlockConfig, ConditionalLayers


class CLVAE(VAE):
    """
    Conditional Latent Variational Autoencoder class.

    This class extends the basic VAE to incorporate conditional layers,
    allowing for conditioning the latent space on additional metadata.

    Args:
        encoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the encoder's fully connected block.
        decoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the decoder's fully connected block.
        conditional_config (Optional[cmmvae.modules.base.FCBlockConfig]):
            Configuration for the conditional layers.
        conditional_paths (Dict[str, str]):
            Mapping of conditional paths for the conditional layers.
        selection_order (Optional[List[str]]):
            Order in which to apply selection of conditionals.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """

    def __init__(
        self,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig,
        conditional_config: Optional[FCBlockConfig],
        conditional_paths: Dict[str, str],
        selection_order: Optional[List[str]] = None,
        **encoder_kwargs
    ):
        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **encoder_kwargs,
        )

        if conditional_config:
            self.conditionals = ConditionalLayers(
                conditional_paths=conditional_paths,
                fc_block_config=conditional_config,
                selection_order=selection_order,
            )
        else:
            self.conditionals = None

    def after_reparameterize(
        self, z: torch.Tensor, metadata: pd.DataFrame
    ) -> torch.Tensor:
        """
        Modify the latent variable after reparameterization
            by applying conditional layers.

        If conditional layers are defined, they will be applied to
            the latent variable `z` using the provided `metadata`.

        Args:
            z (torch.Tensor): Latent variable of shape (batch_size, n_latent).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            torch.Tensor:
                Processed latent variable after applying conditionals, if any.
        """
        if self.conditionals:
            return self.conditionals(z, metadata)
        # Return the unmodified latent variable
        # if no conditionals are present
        return z
