from collections import OrderedDict
from typing import Any, Union
import torch
import torch.nn as nn
from torch.distributions import Normal
from ._vae import VAE
import pandas as pd
from .mixins.init import HeWeightInitMixIn
from .base import BaseModule
from .base import FCBlock
from sciml.utils.constants import REGISTRY_KEYS as RK
from pathlib import Path

_PATH = Union[Path, str]
    
class CLVAE(VAE, HeWeightInitMixIn, BaseModule):
    
    def __init__(self, use_shared_layers: bool = False, conditional_layer_config: dict[str, _PATH] = {}, cl_block_kwargs: dict[str, Any] = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not conditional_layer_config:
            raise RuntimeError("No dropfilters found")
        
        if use_shared_layers:
            self.shared_layers = nn.ModuleDict({
                key: FCBlock([self.n_latent], **cl_block_kwargs) 
                for key in conditional_layer_config
            })
        
        self.conditional_layers = nn.ModuleDict({
            key: self.get_df_module_dict(cl_path, **cl_block_kwargs) 
            for key, cl_path in conditional_layer_config.items()
        })
        
        self.init_weights()
        self.use_shared_layers = use_shared_layers
    
    def get_df_module_dict(self, cl_path, **kwargs):
        
        with open(cl_path, 'r') as file:
            df = pd.read_csv(file, header=None)
            
        return nn.ModuleDict({
            row[0].replace('.', '_'): FCBlock([self.n_latent], **kwargs) 
            for row in df.itertuples(index=False)
        })
        
    def get_module_inputs(self, batch, **module_input_kwargs):
        return (batch[RK.X], batch[RK.METADATA]), module_input_kwargs
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        for cl in self.conditional_layers:
            mask = self._generate_masks(cl, metadata)
            z = self._forward_masks(z, self.conditional_layers[cl], mask)
            if self.use_shared_layers:
                z_shr = self.shared_layers[cl](z)
                z = z_shr - z
            
        return z
    
    def _generate_masks(self, filter_key: str, metadata: pd.DataFrame):
        values = metadata[filter_key]
        masks = {}
        
        for val in values:
            if val in masks:
                continue
            mask = (val == values[:])
            masks[val] = mask.to_list()
            
        return masks
    
    def _forward_masks(self, z, cls, masks):
        cl_sub_batches = []
        for key, mask in masks.items():
            key = key.replace('.', '_')
            if not key in cls:
                raise RuntimeError(f"{key} not in {cls}")
            module = cls[key]
            x = module(z[mask])
            cl_sub_batches.append(x)
        return torch.cat(cl_sub_batches, dim=0)
    
    def configure_optimizers(self):
        return torch.optim.Adam([
            { 'params': self.encoder.parameters(), 'lr': 1e-3},
            { 'params': self.decoder.parameters(), 'lr': 1e-4},
            { 'params': self.conditional_layers.parameters(), 'lr': 1e-4},
        ])