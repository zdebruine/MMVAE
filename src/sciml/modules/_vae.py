from typing import Any, Callable, Literal, NamedTuple, Optional, Union, Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Distribution

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
        
        self.n_latent = n_latent
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
        z = self.after_reparameterize(z, metadata)
        x_hat = self.decode(z)
        return qz, pz, x_hat
    
    def elbo(self, qz: Distribution, pz: Distribution, x: torch.Tensor, x_hat: torch.Tensor, kl_weight: float):
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
        z_kl_div = kl_divergence(qz, pz).sum(dim=-1).mean()
        
        if x.layout == torch.sparse_csr:
            x = x.to_dense()
            
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.shape[0]
        
        loss = (recon_loss + kl_weight * z_kl_div)  # Compute total loss
        
        return z_kl_div, recon_loss, loss

    def loss(
        self, 
        model_inputs,
        model_outputs,
        kl_weight: float = 1.0
    ):
        """
        Compute the loss for the VAE.

        Args:
            model_inputs (tuple): Inputs to the model.
            model_outputs (tuple): Outputs from the model.
            kl_weight (float, optional): Weight for the KL divergence term. Defaults to 1.0.

        Returns:
            dict: Dictionary containing loss components.
        """
        args, _ = model_inputs
        x, *_ = args
        qz, pz, x_hat = model_outputs
        
        # Compute Evidence Lower Bound (ELBO) loss
        z_kl_div, recon_loss, loss = self.elbo(qz, pz, x, x_hat, kl_weight=kl_weight)
        
        return {
            RK.RECON_LOSS: recon_loss / x.shape[1],
            RK.KL_LOSS: z_kl_div,
            RK.LOSS: loss,
            RK.KL_WEIGHT: kl_weight
        }