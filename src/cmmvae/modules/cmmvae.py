import warnings
import pandas as pd
import torch
import torch.nn.functional as F

from .base import Experts
from .clvae import CLVAE
from .mmvae import MMVAE
from cmmvae.constants import REGISTRY_KEYS as RK



class CMMVAE(MMVAE):
    
    vae: CLVAE
    
    def __init__(self, clvae: CLVAE, experts: Experts):
        super().__init__(
            vae=clvae,
            experts=experts
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        metadata: pd.DataFrame,
        expert_id: str,
        cross_generate: bool = False,
    ):
        shared_x = self.experts[expert_id].encode(x)
        
        qz, pz, z, shared_xhat = self.vae(shared_x, metadata)
        
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
                _, _, _, shared_xhat = self.vae(shared_x, metadata)
                cg_xhats[xhat_expert_id] = self.experts[expert_id].decode(shared_xhat)
        else:
            xhats[expert_id] = self.experts[expert_id].decode(shared_xhat)
            
        return qz, pz, z, xhats, cg_xhats
    
    def loss(
        self,
        x: torch.Tensor,
        expert_id: str,
        qz: torch.distributions.Distribution,
        pz: torch.distributions.Distribution,
        xhats: dict[str, torch.Tensor],
        cg_xhats: dict[str, torch.Tensor],
        kl_weight: float,
    ):

        if x.layout == torch.sparse_csr:
            x = x.to_dense()
            
        loss_dict = self.vae.elbo(qz, pz, x, xhats[expert_id], kl_weight)
        
        cg_losses = {
            f"cross_generation/{cg_expert_id}2{expert_id}": F.mse_loss(cg_xhats[cg_expert_id], x, reduction='mean')
            for cg_expert_id in cg_xhats 
        }
        
        return { **cg_losses, **loss_dict }
        
    @torch.no_grad()
    def get_latent_embeddings(self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str):
        
        x = self.experts[expert_id].encode(x)
        qz, z = self.vae.encode(x)

        return {
            RK.Z: z,
            f"{RK.Z}_{RK.METADATA}": metadata
        }