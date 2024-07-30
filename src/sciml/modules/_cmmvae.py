from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from sciml.modules.base._expert import Experts

from ._vae import VAE
from .base import FCBlock
from .base._conditional_layer import ConditionalLayers

from sciml.utils.constants import REGISTRY_KEYS as RK


class Expert(nn.Module):
    
    def __init__(self, encoder_kwargs, decoder_kwargs):
        
        self.encoder = FCBlock(**encoder_kwargs)
        self.decoder = FCBlock(**decoder_kwargs)
    
    def encode(self, x: torch.Tensor):
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor):
        return self.decoder(x)

class CVAE(VAE):
    
    def __init__(
        self,
        vae_kwargs,
        conditional_kwargs,
    ):
        super().__init__(**vae_kwargs)
        self.conditionals = ConditionalLayers(**conditional_kwargs)
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        return self.conditionals(z, metadata)
    
from ._mmvae import MMVAE

class MMCVAE(MMVAE):
    
    def __init__(
        self,
        cvae: CVAE,
        experts: Experts
    ):
        super().__init__(
            vae=cvae,
            experts=experts
        )


class CMMVAE(CVAE):
    
    def __init__(
        self,
        vae_kwargs: dict[str, Any],
        experts_kwargs: dict[str, dict[str, Any]]
    ):
        super().__init__()
        
        self.experts = nn.ModuleDict({
            expert_id: Expert(**expert_config)
            for expert_id, expert_config in experts_kwargs.items()
        })
        
        self.cvae = VAE(**vae_kwargs)
        
    @property
    def forward_kwargs(self):
        return 
        
    def forward(
        self, 
        x: torch.Tensor,
        metadata: pd.DataFrame,
        expert_id: str,
        compute_loss: bool = True,
        **loss_kwargs
    ):
        shared_x = self.experts[expert_id].encode(x)
        
        qz, pz, z, shared_x_hat = self.cvae(shared_x, metadata)
        
        x_hats = {}
        for expert in self.experts:
            x_hats[expert] = self.experts[expert].decode(shared_x_hat)
            
        if compute_loss:
            return self.loss(x, metadata, qz, pz, x_hats, **loss_kwargs)
        
        return qz, pz, z, x_hats
    
    def cross_generate(self, x, source):
        """
        Perform cross-generation between species.

        Args:
            x (torch.Tensor): Input tensor.
            source (str): Source expert ID.

        Returns:
            torch.Tensor: Reconstructed tensor from the source after cross-generation.
        """
        target = RK.MOUSE if source == RK.HUMAN else RK.HUMAN
        
        _, _, target_x_hat = self(x, source, target=target)
        _, _, source_cross_x_hat = self(target_x_hat[target], target, target=source)
        
        return source_cross_x_hat[source]
    
    def loss(
        self,
        x: torch.Tensor,
        metadata: pd.DataFrame,
        expert_id: str,
        qz: torch.distributions.Distribution,
        pz: torch.distributions.Distribution,
        x_hats: dict[str, torch.Tensor],
        kl_weight: float,
        compute_cross_gen_loss: bool,
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
        