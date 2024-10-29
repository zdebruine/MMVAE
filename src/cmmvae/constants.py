"""
    Constants used in cmmvae for dictionary keys.
"""
from typing import NamedTuple


class REGISTRY_KEYS_NT(NamedTuple):
    """A NamedTuple to store constant keys used in a registry for machine learning models."""

    LOSS: str = "loss"
    """Key for the loss value."""
    HUMAN_LOSS: str = "human_loss"
    """Key for the human loss value."""
    MOUSE_LOSS: str = "mouse_loss"
    """Key for the mouse loss value."""
    RECON_LOSS: str = "recon_loss"
    """Key for the reconstruction loss."""
    KL_LOSS: str = "kl_loss"
    """Key for the Kullback-Leibler divergence loss."""
    SHARED_KL: str = "shared_kl"
    """Key for the shared network Kullback-Leibler divergence loss."""
    KL_WEIGHT: str = "kl_weight"
    """Key for the weight of the KL divergence term."""
    LABELS: str = "labels"
    """Key for the labels in the dataset."""
    PX: str = "px"
    """Key for the probabilistic model p(x)."""
    QZ: str = "qz"
    """Key for the posterior distribution q(z)."""
    PZ: str = "pz"
    """Key for the prior distribution p(z)."""
    QZM: str = "qzm"
    """Key for the mean of the posterior distribution q(z)."""
    QZV: str = "qzv"
    """Key for the variance of the posterior distribution q(z)."""
    Z: str = "z"
    """Key for the latent variable z."""
    Z_STAR: str = "z_star"
    """Key for the target or optimal latent variable z*."""
    X: str = "x"
    """Key for the input data x."""
    X_HAT: str = "xhat"
    """Key for the reconstructed data \\hat{x}."""
    HIDDEN: str = "hidden"
    """Key for the hidden representations in the encoder"""
    Y: str = "Y"
    """Key for the target or output data y."""
    METADATA: str = "metadata"
    """Key for additional metadata."""
    EXPERT: str = "expert"
    """Key for expert information."""
    HUMAN: str = "human"
    """Key for human-related data."""
    MOUSE: str = "mouse"
    """Key for mouse-related data."""
    SHARED: str = "shared"
    """Key for multi-species network."""
    VAE: str = "vae"
    """Key for latent VAE networks"""
    ELBO: str = "elbo"
    """Key for the Evidence Lower Bound (ELBO)."""
    REGISTRY: str = "registry"
    """Key for the registry container."""
    EXPERT_ID: str = "expert_id"
    """Key for the expert identifier."""
    GAN_LOSS: str = "gan_loss"
    """Key for GAN loss"""
    ADV_LOSS: str = "adversarial_loss"
    """Key for the total adversarial loss."""
    ADV_WEIGHT: str = "adverserial_weight"
    """Key for the adversarial weight."""
    UMAP_EMBEDDINGS: str = "umap_embeddings"
    """Key for the h5py file umap embeddings"""
    PREDICT_SAMPLES: str = "data"
    """Key for the h5py file predict samples"""


# Instance of _REGISTRY_KEYS_NT for use in the application
REGISTRY_KEYS = REGISTRY_KEYS_NT()
"""Instance of REGISTRY_KEYS_NT"""
