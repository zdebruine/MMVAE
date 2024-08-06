from typing import Optional, Union
import warnings
import pandas as pd
import torch
import torch.nn as nn

from .base import Experts, FCBlock, FCBlockConfig
from .clvae import CLVAE
from .mmvae import MMVAE
from cmmvae.constants import REGISTRY_KEYS as RK


class CMMVAE(MMVAE):
    
    vae: CLVAE
    
    def __init__(
        self, 
        clvae: CLVAE, 
        experts: Experts,
        adversarials: Union[Optional[FCBlockConfig], list[Optional[FCBlockConfig]]] = None,
    ):
        super().__init__(
            vae=clvae,
            experts=experts
        )
        
        self.adversarials = nn.ModuleList([FCBlock(config) for config in adversarials if config])
    
    def forward(
        self, 
        x: torch.Tensor,
        metadata: pd.DataFrame,
        expert_id: str,
        cross_generate: bool = False,
    ):
        shared_x = self.experts[expert_id].encode(x)
        
        qz, pz, z, shared_xhat, hidden_representations = self.vae(shared_x, metadata)
        
        xhats = {}
        cg_xhats = {}
        
        if cross_generate:
            if self.training:
                warnings.warn("CMMVAE is cross generating during training which could cause gradients to be accumulated for cross generation passes")
                
            for expert in self.experts:
                xhats[expert] = self.experts[expert].decode(shared_xhat)
                
            for xhat_expert_id in xhats:
                if xhat_expert_id == expert_id:
                    continue
                shared_x = self.experts[xhat_expert_id].encode(xhats[xhat_expert_id])
                _, _, _, shared_xhat, _ = self.vae(shared_x, metadata)
                cg_xhats[xhat_expert_id] = self.experts[expert_id].decode(shared_xhat)
        else:
            xhats[expert_id] = self.experts[expert_id].decode(shared_xhat)
            
        return qz, pz, z, xhats, cg_xhats, hidden_representations
        
    @torch.no_grad()
    def get_latent_embeddings(self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str):
        
        x = self.experts[expert_id].encode(x)
        qz, z = self.vae.encode(x)

        return {
            RK.Z: z,
            f"{RK.Z}_{RK.METADATA}": metadata
        }