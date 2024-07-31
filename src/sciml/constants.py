from typing import NamedTuple

    
class _REGISTRY_KEYS_NT(NamedTuple):
    LOSS: str = "loss"
    RECON_LOSS: str = "recon_loss"
    KL_LOSS: str = "kl_loss"
    KL_WEIGHT: str = "kl_weight"
    LABELS: str = "labels"
    PX: str = "px"
    QZ: str = "qz"
    PZ: str = "pz"
    QZM: str = "qzm"
    QZV: str = "qzv"
    Z: str = "z"
    Z_STAR: str = "z_star"
    X: str = "x"
    X_HAT: str = "X_HAT"
    Y: str = "Y"
    METADATA: str = "metadata"
    EXPERT: str = "expert"
    HUMAN: str = "human"
    MOUSE: str = "mouse"
    ELBO: str = "elbo"
    REGISTRY: str = "registry"
    EXPERT_ID: str = "expert_id"

REGISTRY_KEYS = _REGISTRY_KEYS_NT()
