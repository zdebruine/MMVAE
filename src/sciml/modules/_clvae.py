from collections import OrderedDict
from typing import Any, Union
import torch
import torch.nn as nn
from torch.distributions import Normal
from ._vae import VAE
import pandas as pd
import numpy as np
from .mixins.init import HeWeightInitMixIn
from .base import FCBlock, ConditionalLayer, BaseModule
from sciml.utils.constants import REGISTRY_KEYS as RK
from pathlib import Path
_PATH = Union[Path, str]
    
class CLVAE(VAE, HeWeightInitMixIn, BaseModule):
    
    def __init__(self, use_shared_layers: bool = False, conditional_layer_config: dict[str, _PATH] = {}, cl_block_kwargs: dict[str, Any] = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not conditional_layer_config:
            import warnings
            warnings.warn("No dropfilters found")
            conditional_layer_config = {}
        
        if use_shared_layers:
            self.shared_layers = nn.ModuleList({
                FCBlock([self.n_latent], **cl_block_kwargs) 
                for _ in range(len(conditional_layer_config))
            })
        
        self.conditional_layers = nn.ModuleList([
            ConditionalLayer(self.n_latent, batch_key, conditions_path, **cl_block_kwargs)
            for batch_key, conditions_path in conditional_layer_config.items()
        ])
        
        self.init_weights()
        self.use_shared_layers = use_shared_layers
        
    def get_module_inputs(self, batch, **module_input_kwargs):
        return (batch[RK.X], batch[RK.METADATA]), module_input_kwargs
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        for idx, layer in enumerate(self.conditional_layers):
            z_star, metadata = layer(z, metadata)
            if self.use_shared_layers:
                z_shr = self.shared_layers[idx](z)
                z = z_shr - z_star
            else:
                z = z_star
        return z, metadata

    def configure_optimizers(self):
        return torch.optim.Adam([
            { 'params': self.encoder.parameters(), 'lr': 1e-3},
            { 'params': self.decoder.parameters(), 'lr': 1e-3},
            { 'params': self.conditional_layers.parameters(), 'lr': 1e-3},
        ])