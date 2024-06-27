from typing import Callable, Literal, NamedTuple, Optional, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from .base import Encoder, FCBlock

from sciml.utils.constants import REGISTRY_KEYS as RK

class VAE(nn.Module):
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
    
    def __init__(
        self,
        n_in: int,
        encoder_layers: list[int],
        n_latent: int,
        decoder_layers: list[int],
        n_out: int,
        distribution: Union[Literal['ln'], Literal['normal']] = 'normal',
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
    ):
        super().__init__()
        
        # Initialize the encoder
        self.encoder = Encoder(
            n_in=n_in,
            fc_layers=encoder_layers[:-1],
            n_hidden=encoder_layers[-1], 
            n_out=n_latent,
            distribution=distribution,
            return_dist=True,
            **encoder_kwargs,
        )

        # Initialize the decoder
        decoder_layers = [n_latent, *decoder_layers, n_out]
        self.decoder = FCBlock(
            layers=decoder_layers,
            **decoder_kwargs
        )
    
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
    
    def decode(self, z: torch.Tensor):
        """
        Decode the latent variable.

        Args:
            z (torch.Tensor): Latent variable.

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        x_hat = self.decoder(z)
        return x_hat
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: The approximate posterior distribution, prior distribution, and reconstructed input tensor.
        """
        qz, z = self.encode(x)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        x_hat = self.decode(z)
        return qz, pz, x_hat
    
    def elbo(self, qz, pz, x, x_hat, kl_weight: float = 1.0):
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        Args:
            qz (torch.distributions.Distribution): Approximate posterior distribution.
            pz (torch.distributions.Distribution): Prior distribution.
            x (torch.Tensor): Original input tensor.
            x_hat (torch.Tensor): Reconstructed input tensor.
            kl_weight (float, optional): Weight for the KL divergence term. Defaults to 1.0.

        Returns:
            tuple: KL divergence, reconstruction loss, and total loss.
        """
        z_kl_div = kl_divergence(qz, pz).sum(dim=-1)  # Compute KL divergence
        recon_loss = F.mse_loss(x_hat, x)  # Compute reconstruction loss
    
        weighted_kl = kl_weight * z_kl_div  # Weight the KL divergence
        
        loss = torch.mean(recon_loss + weighted_kl)  # Compute total loss
        
        return z_kl_div, recon_loss, loss
