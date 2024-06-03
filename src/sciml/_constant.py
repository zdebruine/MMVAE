from typing import NamedTuple

class _REGISTRY_KEYS_NT(NamedTuple):
    LOSS: str = "loss"
    RECON_LOSS: str = "recon_loss"
    KL_LOSS: str = "kl_loss"
    KL_WEIGHT: str = "kl_weight"
    LABELS: str = "labels"
    QZM: str = "qzm"
    QZV: str = "qzv"
    Z: str = "z"
    X: str = "x"
    X_HAT: str = "x_hat"
    Y: str = "Y"
    METADATA: str = "metadata"
    

REGISTRY_KEYS = _REGISTRY_KEYS_NT()