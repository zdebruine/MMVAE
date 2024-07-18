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
from .base import FCBlock, ConditionalLayer, BaseModule
from sciml.utils.constants import REGISTRY_KEYS as RK
from pathlib import Path
_PATH = Union[Path, str]
    
class CLVAE(VAE, HeWeightInitMixIn, BaseModule):
    
    def __init__(
        self, 
        cl_unique_conditions_paths: dict[str, _PATH] = {}, 
        selection_order: Optional[list[str]] = None,
        conditional_block_kwargs: dict[str, Any] = {}, 
        conditions: dict[str, Any] = {},
        batch_keys: Optional[list[str]] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        selection_fn = 'random' if not selection_order else 'sequential'
        if not selection_fn in ('random', 'sequential'):
            raise ValueError("selection_fn must be 'random' or 'sequential'")
        
        if selection_fn == 'random':
            if selection_order:
                import warnings
                warnings.warn("passed selection_fn 'random' will override supplied selection_order")
            selection_order = list(cl_unique_conditions_paths.keys())
        elif selection_fn == 'sequential':
            if selection_order == None:
                raise ValueError(f"Selection order cannot be none if sequential selection")
        else:
            raise ValueError(f"Selection function is not 'sequential' or 'random': {selection_fn}")
        
        self.conditional_layers = nn.ModuleDict({
            batch_key: ConditionalLayer(self.n_latent, batch_key, conditions_path, **conditional_block_kwargs)
            for batch_key, conditions_path in cl_unique_conditions_paths.items()
        })
        
        self.n_selections = len(selection_order)
        assert self.n_selections == len(self.conditional_layers), f"selection_order must be of the same size as conditonal_layers {self.n_selections} : {len(self.conditional_layers)}"
        self._selection_order = selection_order
        self._selection_fn = selection_fn
        self.batch_keys = selection_order.copy()
        
        for selection in self._selection_order:
            if selection not in self.conditional_layers:
                raise KeyError(f"Supplied selection key in selection_order {selection} is not a conditional layer")
        self.conditions = conditions
        self.init_weights()
        
    def get_module_inputs(self, batch, **module_input_kwargs):
        return (batch[RK.X], batch[RK.METADATA]), module_input_kwargs
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        if self.conditions:
            assert all(condition in self.batch_keys for condition in self.conditions), "condition batch key not found in conditional layers"

        if self._selection_fn == 'random':
            random.shuffle(self._selection_order)
        
        for selection in self._selection_order:
            condition = self.conditions[selection] if self.conditions else None
            z, metadata = self.conditional_layers[selection](z, metadata, condition=condition)
            
        return z

    def configure_optimizers(self):
        return torch.optim.Adam([
            { 'params': self.encoder.parameters(), 'lr': 1e-3},
            { 'params': self.decoder.parameters(), 'lr': 1e-3},
            { 'params': self.conditional_layers.parameters(), 'lr': 1e-3},
        ])
        
    def get_latent_embeddings(self, x: torch.Tensor, metadata: pd.DataFrame, conditions: dict[str, Any] = {}):
        self.conditions = conditions
        self._return_z = True
        qz, z, x_hat = self(x, metadata)
        return { 
            RK.X_HAT: x_hat, f"{RK.X_HAT}_{RK.METADATA}": metadata,
            RK.Z: z, f"{RK.Z}_{RK.METADATA}": metadata 
        }