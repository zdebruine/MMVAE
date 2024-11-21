from typing import Optional

import torch
import pandas as pd

from cmmvae.modules.vae import VAE
from cmmvae.modules.base import FCBlockConfig, ConditionalLayers, ConcatBlockConfig


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
        conditional_config: Optional[FCBlockConfig] = None,
        conditionals_directory: Optional[str] = None,
        conditionals: Optional[list[str]] = None,
        selection_order: Optional[list[str]] = None,
        concat_config: Optional[ConcatBlockConfig] = None,
        **encoder_kwargs
    ):
        conditionals_module = None
        if conditional_config and conditionals and conditionals_directory:
            conditionals_module = ConditionalLayers(
                directory=conditionals_directory,
                conditionals=conditionals,
                fc_block_config=conditional_config,
                selection_order=selection_order,
            )
        else:
            import warnings

            warnings.warn("No conditionals found for vae")

        if selection_order and selection_order[0] == "parallel":
            if not concat_config:
                raise RuntimeError(
                    "Please define concat_config when selection_order = parallel"
                )
            concat_dim = (
                len(conditionals_module.selection_order) * conditional_config.layers[-1]
            )

            decoder_config.layers = [concat_dim] + decoder_config.layers
            decoder_config.activation_fn = [
                concat_config.activation_fn
            ] + decoder_config.activation_fn
            decoder_config.dropout_rate = [
                concat_config.dropout_rate
            ] + decoder_config.dropout_rate
            decoder_config.return_hidden = [
                concat_config.return_hidden
            ] + decoder_config.return_hidden
            decoder_config.use_layer_norm = [
                concat_config.use_layer_norm
            ] + decoder_config.use_layer_norm
            decoder_config.use_batch_norm = [
                concat_config.use_batch_norm
            ] + decoder_config.use_batch_norm

        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **encoder_kwargs,
        )

        self.conditionals = conditionals_module

    def after_reparameterize(
        self, z: torch.Tensor, metadata: pd.DataFrame, **kwargs
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
            return self.conditionals(z, metadata, **kwargs)
        # Return the unmodified latent variable
        # if no conditionals are present
        return z
