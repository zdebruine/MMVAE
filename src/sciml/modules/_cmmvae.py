from dataclasses import dataclass
import torch
import torch.nn as nn
import pandas as pd

@dataclass
class CMMVAEResults:
    x_hat: torch.Tensor
    z: torch.Tensor
    metadata: pd.DataFrame

class CMMVAE(nn.Module):
    
    def __init__(
        self,
    ):
        super().__init__()
        
    def forward(
        self, 
        x: torch.Tensor, 
        metadata: pd.DataFrame, 
        expert_id: str
    ) -> CMMVAEResults:
        """Forward pass through model"""
        
        
        
        return CMMVAEResults(
            x_hat=x_hat,
            z=z,
            metadata=metadata
        )