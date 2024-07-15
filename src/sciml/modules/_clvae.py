from collections import OrderedDict
from typing import Any, Union
import torch
import torch.nn as nn
from torch.distributions import Normal
from ._vae import VAE
import pandas as pd
import numpy as np
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
            import warnings
            warnings.warn("No dropfilters found")
            conditional_layer_config = {}
        
        if use_shared_layers:
            self.shared_layers = nn.ModuleDict({
                key: FCBlock([self.n_latent], **cl_block_kwargs) 
                for key in conditional_layer_config
            })
        
        self.conditions_module_dict = nn.ModuleDict({
            condition: self.get_condition_module_dict(conditions_path, **cl_block_kwargs) 
            for condition, conditions_path in conditional_layer_config.items()
        })
        
        self.init_weights()
        self.use_shared_layers = use_shared_layers
    
    def get_condition_module_dict(self, conditions_path, **kwargs):
        
        with open(conditions_path, 'r') as file:
            condition_df = pd.read_csv(file, header=None)
            
        return nn.ModuleDict({
            row[0].replace('.', '_'): FCBlock([self.n_latent], **kwargs) 
            for row in condition_df.itertuples(index=False)
        })
        
    def get_module_inputs(self, batch, **module_input_kwargs):
        return (batch[RK.X], batch[RK.METADATA]), module_input_kwargs
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        for condition in self.conditions_module_dict:
            z, metadata = self.forward_condition(z, metadata, condition)
        return z, metadata
    
    # def _generate_masks(self, filter_key: str, metadata: pd.DataFrame):
    #     # values contains all values of filter_key from batch  
    #     values = metadata[filter_key]
    #     masks = {}
        
    #     for val in values:
    #         if val not in masks:
    #             masks[val] = (val == values[:]).to_list()
    #     return masks
    
    # def _forward_masks(self, z: torch.Tensor, condition: str, metadata: pd.DataFrame, masks: list[pd.DataFrame]):
        
    #     layers = self.conditions_module_dict[condition]
    #     cl_sub_batches = []
    #     reordered_metadata = []
    #     original_indices = []

    #     for key, mask in masks.items():
    #         key = key.replace('.', '_')
    #         if not key in layers:
    #             raise RuntimeError(f"{key} not in {layers}")
            
    #         cl_sub_batches.append(layers[key](z[mask]))
    #         reordered_metadata.append(metadata[mask])
    #         original_indices.append(mask[mask].stack().index)

    #         z = torch.cat(cl_sub_batches, dim=0)
    #         metadata = pd.concat(reordered_metadata).reset_index(drop=True)
        
    #     return z, metadata
    
    def forward_condition(self, z: torch.Tensor, metadata: pd.DataFrame, condition: str):
        # get the modules per condition expression
        condition_module_dict = self.conditions_module_dict[condition]
        # get the keys of the module's needed from metadata
        batch_condition_keys = metadata[condition]
        
        conditions = {}
        tensors_array = []
        indices_array = []
        
        for condition_key in batch_condition_keys:
            if condition_key not in conditions:
                key = condition_key.replace('.', '_')
                if not key in condition_module_dict:
                    raise RuntimeError(f"{condition_key} not in {condition_module_dict}")
                conditions[condition_key] = key
                
                module = condition_module_dict[key]
                mask = (condition_key == batch_condition_keys[:])
                
                indices_tensor = torch.tensor(mask[mask].index.to_list(), device=z.device)
                
                micro_z = z[indices_tensor]
                micro_z_star = module(micro_z)
                
                indices_array.append(indices_tensor)
                tensors_array.append(micro_z_star)
        
        shuffled_indices = torch.cat(indices_array)
        z_star_shuffled = torch.cat(tensors_array)
        z_star = z_star_shuffled[shuffled_indices]
        
        return z_star, metadata

    def configure_optimizers(self):
        return torch.optim.Adam([
            { 'params': self.encoder.parameters(), 'lr': 1e-3},
            { 'params': self.decoder.parameters(), 'lr': 1e-3},
            { 'params': self.conditions_module_dict.parameters(), 'lr': 1e-3},
        ])