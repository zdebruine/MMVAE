from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from sciml.modules.base._experts import Experts

from ._vae import VAE
from .base import FCBlock, FCBlockConfig
from .base._conditionals import ConditionalLayers

from sciml.constants import REGISTRY_KEYS as RK


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