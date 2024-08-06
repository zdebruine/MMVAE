import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Distribution
from typing import List, Tuple, Dict, Union, Literal

from .base import Encoder, FCBlock, FCBlockConfig
from cmmvae.constants import REGISTRY_KEYS as RK



class BaseVAE(nn.Module):
    """
    Base class for a Variational Autoencoder (VAE).

    This class implements a basic structure for a VAE with configurable encoder and decoder architectures.

    Args:
        encoder (Encoder): The encoder model.
        decoder (nn.Module): The decoder model.
    """
    
    def __init__(self, encoder: Encoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x: torch.Tensor) -> Tuple[Distribution, torch.Tensor, List[torch.Tensor]]:
        """
        Encode the input tensor into a latent representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).

        Returns:
            tuple:
                - qz (Distribution): The approximate posterior distribution over the latent space.
                - z (torch.Tensor): Sampled latent variable from qz.
                - hidden_representations (List[torch.Tensor]): List of hidden representations from the encoder.
        """
        qz, z, hidden_representations = self.encoder(x)
        return qz, z, hidden_representations
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent variable to reconstruct the input.

        Args:
            z (torch.Tensor): Latent variable of shape (batch_size, n_latent).

        Returns:
            torch.Tensor: Reconstructed input tensor of shape (batch_size, n_out).
        """
        xhat = self.decoder(z)
        return xhat
    
    def after_reparameterize(self, z: torch.Tensor, metadata: pd.DataFrame) -> torch.Tensor:
        """
        Optional processing after reparameterization.

        This method can be overridden by subclasses to apply any additional processing to
        the latent variable after reparameterization but before decoding.

        Args:
            z (torch.Tensor): Latent variable.
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            torch.Tensor: Processed latent variable.
        """
        return z
        
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame) -> Tuple[Distribution, Distribution, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            tuple:
                - qz (Distribution): Approximate posterior distribution over the latent space.
                - pz (Distribution): Prior distribution over the latent space.
                - z (torch.Tensor): Sampled latent variable.
                - xhat (torch.Tensor): Reconstructed input tensor.
                - hidden_representations (List[torch.Tensor]): Hidden representations from the encoder.
        """
        qz, z, hidden_representations = self.encode(x)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        z = self.after_reparameterize(z, metadata)
        xhat = self.decode(z)
        return qz, pz, z, xhat, hidden_representations
    
    def elbo(self, qz: Distribution, pz: Distribution, x: torch.Tensor, xhat: torch.Tensor, kl_weight: float) -> Dict[str, torch.Tensor]:
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        The ELBO loss is the sum of the reconstruction loss and the KL divergence,
        weighted by a given factor.

        Args:
            qz (torch.distributions.Distribution): Approximate posterior distribution.
            pz (torch.distributions.Distribution): Prior distribution.
            x (torch.Tensor): Original input tensor of shape (batch_size, n_in).
            xhat (torch.Tensor): Reconstructed input tensor of shape (batch_size, n_out).
            kl_weight (float): Weight for the KL divergence term.

        Returns:
            dict: Dictionary containing the following keys and values:
                - RK.RECON_LOSS: Reconstruction loss normalized by the number of elements.
                - RK.KL_LOSS: Mean KL divergence.
                - RK.LOSS: Total loss.
                - RK.KL_WEIGHT: KL weight.
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
    def get_latent_embeddings(self, x: torch.Tensor, metadata: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Obtain latent embeddings from the input data.

        This method returns the latent embeddings and associated metadata for the input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            dict: Dictionary containing the following keys and values:
                - RK.Z: Latent embeddings.
                - f"{RK.Z}_{RK.METADATA}": Metadata.
        """
        _, z, _ = self.encode(x)
        
        return {
            RK.Z: z,
            f"{RK.Z}_{RK.METADATA}": metadata
        }

class VAE(BaseVAE):
    """
    Variational Autoencoder (VAE) with configurable encoder and decoder blocks.

    This class extends the BaseVAE to utilize specific configurations for the encoder and decoder.

    Args:
        encoder_config (FCBlockConfig): Configuration for the encoder's fully connected block.
        decoder_config (FCBlockConfig): Configuration for the decoder's fully connected block.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """
    
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