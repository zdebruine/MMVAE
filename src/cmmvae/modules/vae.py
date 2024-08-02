from typing import Any, Callable, Literal, NamedTuple, Optional, Union, Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Distribution

from .base import Encoder, FCBlock, FCBlockConfig

from cmmvae.constants import REGISTRY_KEYS as RK
        

class BaseVAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    This class implements a VAE with configurable encoder and decoder architectures.
    
    Args:
        n_in (int): Number of input features.
        encoder_layers (list[int]): List of layer sizes for the encoder.
        n_latent (int): Size of the latent space.
        decoder_layers (list[int]): List of layer sizes for the decoder.
        n_out (int): Number of output features.
        distribution (Union[Literal['ln'], Literal['normal']], optional): Type of distribution for the latent variables. Defaults to 'normal'.
        encoder_kwargs (dict, optional): Additional keyword arguments for the encoder. Defaults to an empty dict.
        decoder_kwargs (dict, optional): Additional keyword arguments for the decoder. Defaults to an empty dict.
    """
    
    def __init__(self, encoder: Encoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    
    def encode(self, x: torch.Tensor):
        """
        Encode the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: The approximate posterior distribution and sampled latent variable.
        """
        qz, z = self.encoder(x)
        return qz, z
    
    def decode(self, x: torch.Tensor):
        """
        Decode the latent variable.

        Args:
            z (torch.Tensor): Latent variable.

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        xhat = self.decoder(x)
        return xhat
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame):
        return z
        
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: The approximate posterior distribution, prior distribution, and reconstructed input tensor.
        """
        qz, z = self.encode(x)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        x = self.after_reparameterize(z, metadata)
        xhat = self.decode(x)
        return qz, pz, z, xhat
    
    def elbo(self, qz: Distribution, pz: Distribution, x: torch.Tensor, xhat: torch.Tensor, kl_weight: float):
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        Args:
            qz (torch.distributions.Distribution): Approximate posterior distribution.
            pz (torch.distributions.Distribution): Prior distribution.
            x (torch.Tensor): Original input tensor.
            xhat (torch.Tensor): Reconstructed input tensor.
            kl_weight (float, optional): Weight for the KL divergence term. Defaults to 1.0.

        Returns:
            tuple: KL divergence, reconstruction loss, and total loss.
        """
        z_kl_div = kl_divergence(qz, pz).sum(dim=-1)
        
        if x.layout == torch.sparse_csr:
            x = x.to_dense()
            
        recon_loss = F.mse_loss(xhat, x, reduction='sum')
        
        loss = (recon_loss + kl_weight * z_kl_div.mean())  # Compute total loss
        
        return {
            RK.RECON_LOSS: recon_loss / x.numel(),
            RK.KL_LOSS: z_kl_div.mean(),
            RK.LOSS: loss,
            RK.KL_WEIGHT: kl_weight
        }
        
    @torch.no_grad()
    def get_latent_embeddings(self, x: torch.Tensor, metadata: pd.DataFrame):

        _, z = self.encode(x)
        
        return {
            RK.Z: z,
            f"{RK.Z}_{RK.METADATA}": metadata
        }
        
class VAE(BaseVAE):
    
    def __init__(
        self,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig,
        **encoder_kwargs,
    ):
        super(VAE, self).__init__(
            encoder=Encoder(
                fc_block_config=encoder_config,
                return_dist=True,
                **encoder_kwargs), 
            decoder=FCBlock(decoder_config)
        )
        
            