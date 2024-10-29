import torch
import torch.nn as nn
import torch.nn.functional as F
from cmmvae.modules.clvae import CLVAE
from torch.distributions import kl_divergence, Distribution
from cmmvae.constants import REGISTRY_KEYS as RK


class MOE_CLVAE(CLVAE):
    """
    Mixture of Experts Conditional Latent Variational Autoencoder class.

    This class extends the CLVAE class, used to create human and
    mouse specific variational autoencoders.

    Args:
        encoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the encoder's fully connected block.
        decoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the decoder's fully connected block.
        conditional_config (Optional[cmmvae.modules.base.FCBlockConfig]):
            Configuration for the conditional layers.
        conditional_paths (Dict[str, str]):
            Mapping of conditional paths for the conditional layers.
        selection_order (Optional[List[str]]):
            Order in which to apply selection of conditionals.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """

    def __init__(self, id: str, **clvae_kwargs):
        super().__init__(**clvae_kwargs)
        self.id = id

    def kl_loss(
        self,
        qz: Distribution,
        pz: Distribution,
        kl_weight: float,
        **kwargs,
    ):
        z_kl_div = kl_divergence(qz, pz).sum(dim=-1)
        return kl_weight * z_kl_div.mean()
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        xhat: torch.Tensor,
        **kwargs,
    ):
        if x.layout == torch.sparse_csr:
            x = x.to_dense()
        return F.mse_loss(xhat, x, reduction="sum")

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
        z_kl_div = self.kl_loss(qz, pz, kl_weight)

        recon_loss = self.reconstruction_loss(x, xhat)

        loss = recon_loss + z_kl_div

        return {
            RK.RECON_LOSS: recon_loss / x.numel(),
            RK.KL_LOSS: z_kl_div,
            RK.LOSS: loss,
            RK.KL_WEIGHT: kl_weight,
        }


class ExpertVAEs(nn.ModuleDict):
    """
    Container to store human and mouse specific VAEs.

    Args:
        vaes (list[cmmvae.modules.MOE_CLVAE]):
            List of Expert VAEs.

    Attributes:
        labels (dict[str, int]):
            Dictionary of vae.id's to their integer representation.
    """

    def __init__(self, vaes: list[MOE_CLVAE]):
        super().__init__({vae.id: vae for vae in vaes if isinstance(vae, MOE_CLVAE)})
        self.labels = {key: i for i, key in enumerate(self.keys())}
