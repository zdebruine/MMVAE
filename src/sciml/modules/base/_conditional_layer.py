import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
        for group_name, group_df in metadata.groupby(self.batch_key):
            mask = torch.zeros(len(metadata), dtype=int, device=device)
            mask[group_df.index] = 1
            binary_masks[group_name] = mask
        return binary_masks
    
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame):
        
        masks = self.mask(metadata, device=x.device)
        
        x_hat_partials = []
        for condition_key, mask in masks.items():
            masked_x = x * mask.unsqueeze(1)
            x_hat_partial = self.condition_modules[condition_key.replace('.', '_')](masked_x)
            x_hat_partials.append(x_hat_partial)
        return sum(x_hat_partials), metadata