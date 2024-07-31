from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Experts
from ._clvae import CLVAE
from ._mmvae import MMVAE
from sciml.constants import REGISTRY_KEYS as RK



class CMMVAE(MMVAE):
    
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
    ):
        shared_x = self.experts[expert_id].encode(x)
        
        qz, pz, z, shared_x_hat = self.vae(shared_x, metadata)
        
        x_hats = {}
        for expert in self.experts:
            x_hats[expert] = self.experts[expert].decode(shared_x_hat)
        
        return qz, pz, z, x_hats
    
    def loss(
        self,
        x: torch.Tensor,
        expert_id: str,
        qz: torch.distributions.Distribution,
        pz: torch.distributions.Distribution,
        x_hats: dict[str, torch.Tensor],
        kl_weight: float,
        compute_cross_gen_loss: bool = False,
    ):

        if x.layout == torch.sparse_csr:
            x = x.to_dense()
            
        z_kl_div, recon_loss, loss = self.elbo(qz, pz, x, x_hats[expert_id], kl_weight)
        
        cross_gen_loss = {}
        if compute_cross_gen_loss:
            if self.training:
                raise RuntimeError("Cannot compute cross gen loss in training mode")
            
            cross_expert_x_hat = self.cross_generate(x, expert_id)
            
            cross_loss = F.mse_loss(cross_expert_x_hat, x, reduction='mean')
            cross_gen_loss[f"cross_gen_loss/{expert_id}"] = cross_loss
        
        return {
            **cross_gen_loss,
            RK.RECON_LOSS: recon_loss,
            RK.KL_LOSS: z_kl_div,
            RK.LOSS: loss,
            RK.KL_WEIGHT: kl_weight
        }
        
    @torch.no_grad()
    def get_latent_embeddings(self, x: torch.Tensor, metadata: pd.DataFrame, expert_id: str):
        
        x = self.experts[expert_id].encode(x)
        qz, z = self.vae.encode(x)
        
        return {
            RK.Z: z,
            f"{RK.Z}_{RK.METADATA}": metadata
        }