"""
    This module holds torch.nn.Modules with sole responsibility
    on neural network creation/forward pass.
"""
from cmmvae.modules import base
from cmmvae.modules.vae import VAE
from cmmvae.modules.clvae import CLVAE
from cmmvae.modules.moe_clvae import MOE_CLVAE, ExpertVAEs
from cmmvae.modules.cmmvae import CMMVAE
from cmmvae.modules.moe_cmmvae import MOE_CMMVAE

__all__ = [
    "base",
    "CLVAE",
    "MOE_CLVAE",
    "ExpertVAEs",
    "CMMVAE",
    "MOE_CMMVAE",
    "VAE",
]
