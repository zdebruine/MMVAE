import torch
import torch.nn as nn
from .mixins.vae import VAEMixIn
from .mixins.init import HeWeightInitMixIn

class BasicVAE(VAEMixIn, HeWeightInitMixIn, nn.Module):
    
    def __init__(
        self,
        encoder_layers = [60664, 1024, 512], 
        latent_dim = 256, 
        decoder_layers = [512, 1024, 60664],
        use_he_init = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.build()
        
        if use_he_init:
            self.init_weights()

        
    def build(self):
        """Initializes attributes encoder, decoder, fc_mean, fc_var"""
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.fc_mean = self.build_mean()
        self.fc_var = self.build_var()
        
    def build_encoder(self):
        layers = []
        n_in = self.encoder_layers[0]
        for n_out in self.encoder_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def build_decoder(self):
        layers = []
        n_in = self.latent_dim
        for n_out in self.decoder_layers:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def _build_mean_var(self):
        return nn.Linear(self.encoder_layers[-1], self.latent_dim)
    
    build_mean = _build_mean_var
    build_var = _build_mean_var
    
class VAE(VAEMixIn, HeWeightInitMixIn, nn.Module):
    
    def __init__(self):
        
        self.encoder = nn.Sequential(
            nn.Linear(60664, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
        )