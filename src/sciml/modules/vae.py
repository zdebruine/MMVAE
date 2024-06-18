from typing import Union
import torch
import torch.nn as nn
from .mixins.vae import VAEMixIn
from .mixins.init import HeWeightInitMixIn
from ._lightning import LightningSequential


class VAE(VAEMixIn, HeWeightInitMixIn, nn.Module):
    
    def __init__(
        self,
        encoder: LightningSequential,
        decoder: LightningSequential,
        mean: nn.Linear,
        var: nn.Linear,
        use_he_init: bool = True,
        optimizer: bool = None,
        fc_mean_lr: float = 1e-4,
        fc_var_lr: float = 1e-4,
        encoder_lr: Union[float, list[float]] = 1e-3,
        decoder_lr: Union[float, list[float]] = 1e-4,
    ):
        super().__init__()
        self.fc_mean = mean
        self.fc_var = var
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.fc_mean_lr = fc_mean_lr
        self.fc_var_lr = fc_var_lr
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.latent_dim = self.fc_mean.out_features
        if use_he_init:
            self.init_weights()
            
    def configure_optimizers(self):
        
        if isinstance(self.encoder_lr, list):
            assert len(self.encoder_lr) == len(self.encoder)
            group = [{'params': getattr(self, name).parameters(), 'lr': getattr(self, f"{name}_lr")} for name in ('fc_mean', 'fc_var', 'encoder', 'decoder')]
            return torch.optim.Adam(group)