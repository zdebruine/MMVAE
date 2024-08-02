from typing import Optional
import torch
import torch.nn as nn
import pandas as pd

from ._fc_block import FCBlock, FCBlockConfig



class ConditionalLayer(nn.Module):
    
    def __init__(self, batch_key: str, conditions_path: str, fc_block_config: FCBlockConfig):
        super(ConditionalLayer, self).__init__()
        
        self.batch_key = batch_key
        self.unique_conditions = { self.format_condition_key(condition) for condition in pd.read_csv(conditions_path, header=None)[0] }
        
        self.conditions = nn.ModuleDict({
            condition: FCBlock(fc_block_config) 
            for condition in self.unique_conditions
        })
        
    def format_condition_key(self, condition: str):
        return condition.replace('.', '_')
    
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, condition: Optional[str] = None):
        
        if condition:
            return self.conditions[condition](x), metadata
        
        active_conditions = set()
        xhat = torch.zeros_like(x)
        
        for condition, group_df in metadata.groupby(self.batch_key):
            condition = self.format_condition_key(condition)
            mask = torch.zeros(len(metadata), dtype=int, device=x.device)
            mask[group_df.index] = 1
            xhat = xhat + self.conditions[condition](x * mask.unsqueeze(1))
            active_conditions.add(condition)
            
        self.active_condition_modules = active_conditions
        
        return xhat
    
    
class ConditionalLayers(nn.Module):
    
    def __init__(
        self,
        conditional_paths: dict[str, str],
        fc_block_config: FCBlockConfig,
        selection_order: Optional[list[str]] = None,
    ):
        super(ConditionalLayers, self).__init__()
        
        if not selection_order:
            selection_order = list(conditional_paths.keys())
            self.shuffle_selection_order = True
        else:
            self.shuffle_selection_order = False
            
        self.selection_order = torch.arange(0, len(selection_order), dtype=torch.int32, requires_grad=False)
        
        self.layers: list[ConditionalLayer] = nn.ModuleList([
            ConditionalLayer(batch_key, conditional_paths[batch_key], fc_block_config)
            for batch_key in selection_order
        ])
        
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, conditions: Optional[dict[str, str]] = None):
        order = self.selection_order
        
        if self.shuffle_selection_order:
            permutation = torch.randperm(self.selection_order.size(0))
            order = order[permutation]
            
        for idx in order:
            x = self.layers[idx](x, metadata)
        
        return x
        
        