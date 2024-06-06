import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from . import utils

from sciml._constant import REGISTRY_KEYS as RK

class AbstractVAEBuilderMixIn:
    
    def build_encoder(self) -> nn.Module:
        """Returned nn.Module to be used as encoder"""

    def build_decoder(self) -> nn.Module:
        """Returned nn.Module to be used as decoder"""
    
    def build_mean(self) -> nn.Module:
        """Returned nn.Module to be used as mean"""
        
    def build_var(self) -> nn.Module:
        """Returned nn.Module to be used as var"""

class VAEBuilderMixIn(AbstractVAEBuilderMixIn):
    
    def build_encoder(self):
        layers = []
        n_in = self.hparams.encoder_layers[0]
        for n_out in self.hparams.encoder_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def _build_mean_var(self):
        return nn.Linear(self.hparams.encoder_layers[-1], self.hparams.latent_dim)
    
    build_mean = _build_mean_var
    build_var = _build_mean_var
    
    def build_decoder(self):
        layers = []
        n_in = self.hparams.latent_dim
        for n_out in self.hparams.decoder_layers:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)

