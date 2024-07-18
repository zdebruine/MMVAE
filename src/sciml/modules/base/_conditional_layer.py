from typing import Literal, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import random

from ._fc_block import FCBlock



class ConditionalLayer(nn.Module):
    def __init__(self, input_size, batch_key, conditions_path, **kwargs):
        super(ConditionalLayer, self).__init__()
        
        conditions_df = pd.read_csv(conditions_path, header=None)

        self.condition_modules = nn.ModuleDict({
            str(condition).replace('.', '_'): FCBlock(layers=[input_size], **kwargs) 
            for condition in conditions_df[0]
        })
        
        self.n_conditions = len(conditions_df)
        self.batch_key = batch_key
        
    def mask(self, metadata, device=torch.dtype):
        binary_masks = {}
        for condition, group_df in metadata.groupby(self.batch_key):
            mask = torch.zeros(len(metadata), dtype=int, device=device)
            mask[group_df.index] = 1
            binary_masks[condition] = mask
        return binary_masks
    
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, condition: Optional[str] = None):
        
        if condition:
            return self.forward_condition(x, condition), metadata
        
        masks = self.mask(metadata, device=x.device)
        
        x_hat_partials = []
        for condition, mask in masks.items():
            masked_x = x * mask.unsqueeze(1)
            x_hat_partial = self.forward_condition(masked_x, condition)
            x_hat_partials.append(x_hat_partial)
        return sum(x_hat_partials), metadata
    
    def forward_condition(self, x: torch.Tensor, condition: str):
        condition = condition.replace('.', '_')
        return self.condition_modules[condition](x)


class DynamicConditionalLayer(ConditionalLayer):
    
    def __init__(self, input_size, batch_key, **block_kwargs):
        super(ConditionalLayer, self).__init__()

        self.input_size = input_size
        self.condition_modules = nn.ModuleDict()
        self.batch_key = batch_key
        self._block_kwargs = block_kwargs
        self.new_modules = []
        
    def get_condition_key(self, condition: str):
        return condition.replace('.', '_')
    
    def forward_condition(self, x: torch.Tensor, condition: str):
        condition_key = self.get_condition_key(condition)
        if not condition in self.condition_modules:
            module = FCBlock(layers=[self.input_size], **self._block_kwargs).to(x.device)
            self.condition_modules.add_module(condition_key, module)
            self.new_modules.append(module)
            
        return super().forward_condition(x, condition_key)
        
        