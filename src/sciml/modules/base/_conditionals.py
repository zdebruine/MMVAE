from typing import Optional
import torch
import torch.nn as nn
import pandas as pd

from ._fc_block import FCBlock, FCBlockConfig



class ConditionalLayer(nn.Module):
    
    def __init__(self, batch_key: str, conditions_path: str, fc_block_config: FCBlockConfig):
        super(ConditionalLayer, self).__init__()
        
        self.batch_key = batch_key
        conditions_df = pd.read_csv(conditions_path, header=None)
        
        self.conditions = nn.ModuleDict({
            self.format_condition_key(condition): FCBlock(fc_block_config) 
            for condition in conditions_df[0]
        })
        
    def format_condition_key(self, condition: str):
        return condition.replace('.', '_')
    
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, condition: Optional[str] = None):
        
        if condition:
            return self.forward_condition(x, condition), metadata
        
        x_hat = torch.zeros_like(x)
        for condition, group_df in metadata.groupby(self.batch_key):
            mask = torch.zeros(len(metadata), dtype=int, device=x.device)
            mask[group_df.index] = 1
            x_hat = x_hat + self.forward_condition(x * mask.unsqueeze(1), condition)
            
        return x_hat, metadata
    
    def forward_condition(self, x: torch.Tensor, condition: str):
        condition = condition.replace('.', '_')
        return self.conditions[condition](x)
    
    
class ConditionalLayers(nn.Module):
    
    def __init__(
        self,
        conditional_paths: dict[str, str],
        fc_block_config: FCBlockConfig,
        selection_order: Optional[list[str]] = None,
    ):
        super(ConditionalLayers, self).__init__()
        
        self.is_random = bool(selection_order)
        
        if self.is_random:
            selection_order = list(conditional_paths.keys())
        
        self.selection_order: torch.Tensor = torch.arange(0, len(selection_order), dtype=torch.int32, requires_grad=False)
        
        self.conditionals = nn.ModuleList([
            ConditionalLayer(batch_key, conditional_paths[batch_key], fc_block_config)
            for batch_key in selection_order
        ])
        
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, conditions: Optional[dict[str, str]] = None):
        order = self.selection_order
        
        if self.is_random:
            permutation = torch.randperm(self.selection_order.size(0))
            order = order[permutation]
            
        for idx in order:
            x, metadata = self.conditionals[idx](x, metadata)
        
        return x, metadata
        
        