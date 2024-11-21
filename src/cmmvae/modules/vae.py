import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Distribution

from cmmvae.modules import base
from cmmvae.constants import REGISTRY_KEYS as RK


class BaseVAE(nn.Module):
    """
    Base class for a Variational Autoencoder (VAE).

    This class implements a basic structure for a VAE
    with configurable encoder and decoder architectures.

    Args:
        encoder (cmmvae.modules.baseEncoder): The encoder model.
        decoder (nn.Module): The decoder model.
    """

    def __init__(self, encoder: base.Encoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: torch.Tensor, **kwargs):
        """
        Encode the input tensor into a latent representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).

        Returns:
            tuple:
                - qz (Distribution): The approximate posterior distribution
                    over the latent space.
                - z (torch.Tensor): Sampled latent variable from qz.
                - hidden_representations (List[torch.Tensor]):
                    List of hidden representations from the encoder.
        """
        qz, z, hidden_representations = self.encoder(x)
        return qz, z, hidden_representations

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode the latent variable to reconstruct the input.

        Args:
            z (torch.Tensor): Latent variable of
                shape (batch_size, n_latent).

        Returns:
            torch.Tensor: Reconstructed input tensor of
                shape (batch_size, n_out).
        """
        xhat = self.decoder(z)
        return xhat

    def after_reparameterize(
        self, z: torch.Tensor, metadata: pd.DataFrame, **kwargs
    ) -> torch.Tensor:
        """
        Optional processing after reparameterization.

        This method can be overridden by subclasses to apply any
        additional processing to the latent variable
        after reparameterization but before decoding.

        Args:
            z (torch.Tensor): Latent variable.
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            torch.Tensor: Processed latent variable.
        """
        return z

    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, **kwargs):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            tuple:
                - qz (Distribution): Approximate posterior distribution
                    over the latent space.
                - pz (Distribution): Prior distribution over the latent space.
                - z (torch.Tensor): Sampled latent variable.
                - xhat (torch.Tensor): Reconstructed input tensor.
                - hidden_representations (List[torch.Tensor]):
                    Hidden representations from the encoder.
        """
        qz, z, hidden_representations = self.encode(x, **kwargs)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        z = self.after_reparameterize(z, metadata, **kwargs)
        xhat = self.decode(z, **kwargs)
        return qz, pz, z, xhat, hidden_representations

    def elbo(
        self,
        qz: Distribution,
        pz: Distribution,
        x: torch.Tensor,
        xhat: torch.Tensor,
        kl_weight: float,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        The ELBO loss is the sum of the reconstruction loss
        and the KL divergence, weighted by a given factor.

        Args:
            qz (Distribution): Approximate posterior distribution.
            pz (Distribution): Prior distribution.
            x (torch.Tensor): Original input tensor of
                shape (batch_size, n_in).
            xhat (torch.Tensor): Reconstructed input tensor of
                shape (batch_size, n_out).
            kl_weight (float): Weight for the KL divergence term.

        Returns:
            dict: Dictionary containing the following keys and values:
                - RK.RECON_LOSS: Reconstruction loss
                    normalized by the number of elements.
                - RK.KL_LOSS: Mean KL divergence.
                - RK.LOSS: Total loss.
                - RK.KL_WEIGHT: KL weight.
        """
        z_kl_div = kl_divergence(qz, pz)
        z_kl_div = z_kl_div.sum(dim=-1)
        z_kl_div = z_kl_div.mean()

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        recon_loss = F.mse_loss(xhat, x, reduction="sum")
        # recon_loss = F.mse_loss(xhat, x, reduction="none")
        # recon_loss = recon_loss.sum(dim=1)

        loss = recon_loss + (kl_weight * z_kl_div)
        # loss = torch.mean(z_kl_div * kl_weight + recon_loss)

        recon_loss = recon_loss / x.numel()
        # recon_loss = recon_loss.mean()

        return {
            RK.LOSS: loss,
            RK.RECON_LOSS: recon_loss,
            RK.KL_LOSS: z_kl_div,
            RK.KL_WEIGHT: kl_weight,
        }

    @torch.no_grad()
    def get_latent_embeddings(
        self, x: torch.Tensor, metadata: pd.DataFrame, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Obtain latent embeddings from the input data.

        This method returns the latent embeddings and
        associated metadata for the input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_in).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            dict: Dictionary containing the following keys and values:
                - RK.Z: Latent embeddings.
                - f"{RK.Z}_{RK.METADATA}": Metadata.
        """
        _, z, _ = self.encode(x)

        return {RK.Z: z, f"{RK.Z}_{RK.METADATA}": metadata}


class VAE(BaseVAE):
    """
    Variational Autoencoder (VAE)
    with configurable encoder and decoder blocks.

    This class extends the BaseVAE to utilize
    specific configurations for the encoder and decoder.

    Args:
        encoder_config (cmmvae.modules.baseFCBlockConfig):
            Configuration for the encoder's fully connected block.
        decoder_config (cmmvae.modules.baseFCBlockConfig):
            Configuration for the decoder's fully connected block.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """

    def __init__(
        self,
        encoder_config: base.FCBlockConfig,
        decoder_config: base.FCBlockConfig,
        **encoder_kwargs,
    ):
        super().__init__(
            encoder=base.Encoder(
                fc_block_config=encoder_config, return_dist=True, **encoder_kwargs
            ),
            decoder=base.FCBlock(decoder_config),
        )
