
import torch.nn as nn
from cmmvae.constants import REGISTRY_KEYS as RK

from .vae import VAE
from .base import Experts

class MMVAE(nn.Module):
    """
    Multi-Modal Variational Autoencoder (MMVAE) class.

    This class extends HeWeightInitMixIn and BaseModule, incorporating a VAE and multiple experts for encoding and decoding.
    
    Args:
        vae (VAE): The variational autoencoder module.
        experts (Experts): The experts module containing human and mouse encoders/decoders.
    """
    
    def __init__(
        self,
        vae: VAE,
        experts: Experts
    ):
        super().__init__()
        
        self.vae = vae
        self.experts = experts
        
    def forward(self, x, source: str, target: str = None):
        """
        Forward pass through the MMVAE.

        Args:
            x (torch.Tensor): Input tensor.
            source (str): Source expert ID.
            target (str, optional): Target expert ID. Defaults to None.

        Returns:
            tuple: The latent variables and reconstructed outputs.
        """
        if target:
            experts = {target: self.experts[target]}
        else:
            experts = self.experts
            
        shared_x = self.experts[source].encode(x)  # Encode input using the source expert
        
        qz, pz, shared_x_hat = self.vae(shared_x)  # Pass through the VAE
        
        expert_x_hats = {}
        for expert_id, expert in experts.items():
            xhat = expert.decode(shared_x_hat)  # Decode the shared representation using each expert
            expert_x_hats[expert_id] = xhat
            
        return qz, pz, expert_x_hats