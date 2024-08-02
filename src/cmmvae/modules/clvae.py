from typing import Optional
import torch
import pandas as pd

from .vae import VAE
from .base import FCBlockConfig, ConditionalLayers


class CLVAE(VAE):
    
    def __init__(
        self,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig,
        conditional_config: Optional[FCBlockConfig],
        conditional_paths: dict[str, str],
        selection_order: Optional[list[str]] = None,
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
                selection_order=selection_order)
        else:
            self.conditionals = None
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        if self.conditionals:
            return self.conditionals(z, metadata)