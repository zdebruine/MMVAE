from collections import OrderedDict
import random
from typing import Any, Literal, Optional, Union
import torch
import torch.nn as nn
from torch.distributions import Normal
from ._vae import VAE
import pandas as pd
import numpy as np
from .mixins.init import HeWeightInitMixIn
from .base import FCBlock, DynamicConditionalLayer, BaseModule
from sciml.utils.constants import REGISTRY_KEYS as RK
from pathlib import Path
_PATH = Union[Path, str]

    
class DynamicCLVAE(VAE, HeWeightInitMixIn, BaseModule):
    
    def __init__(
        self, 
        batch_keys: list[str],
        selection_order: Optional[list[str]] = None,
        conditional_block_kwargs: dict[str, Any] = {},
        **kwargs
    ):
        super().__init__(**kwargs)
        selection_fn = 'sequential' if selection_order else 'random'
        if selection_fn == 'random':
            selection_order = batch_keys.copy()
        assert len(batch_keys) == len(selection_order), "selection_order and batch_keys mismatched length"
        assert all(key in selection_order for key in batch_keys), "selection_order contains keys not found in batch_keys"
        
        self.conditional_layers = nn.ModuleDict({
            batch_key: DynamicConditionalLayer(self.n_latent, batch_key, **conditional_block_kwargs)
            for batch_key in batch_keys
        })
        
        self._selection_fn = selection_fn
        self._selection_order = selection_order
        self.batch_keys = batch_keys
    
    def get_module_inputs(self, batch, **module_input_kwargs):
        return (batch[RK.X], batch[RK.METADATA]), module_input_kwargs
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        if self._selection_fn == 'random':
            random.shuffle(self._selection_order)
            
        new_modules = []
        for selection in self._selection_order:
            z, metadata = self.conditional_layers[selection](z, metadata)
            _new_modules = self.conditional_layers[selection].new_modules
            new_modules.extend(_new_modules)
            _new_modules.clear()
        
        self.update_optimzer(new_modules)
        
        return z
    
    def update_optimzer(self, new_modules: list[nn.Module]):
        for module in new_modules:
            self.optimizer.add_param_group({'params': module.parameters()})
        new_modules.clear()
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([
            { 'params': self.encoder.parameters(), 'lr': 1e-3},
            { 'params': self.decoder.parameters(), 'lr': 1e-3},
            { 'params': self.conditional_layers.parameters(), 'lr': 1e-3},
        ])
        return self.optimizer