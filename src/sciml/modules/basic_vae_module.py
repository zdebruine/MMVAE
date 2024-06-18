import torch
import torch.nn as nn
from .vae import VAE
        
        
class BasicVAE(VAE):
    
    def __init__(
        self,
        encoder_layers = [60664, 1024, 512], 
        latent_dim = 256, 
        decoder_layers = [512, 1024, 60664],
        use_he_init = True,
    ):
        super().__init__(
            encoder=self.build_encoder(encoder_layers),
            decoder=self.build_decoder(latent_dim, decoder_layers),
            mean=self.build_mean(encoder_layers, latent_dim),
            var=self.build_var(encoder_layers, latent_dim),
            use_he_init=use_he_init
        )
        
    def build_encoder(self, encoder_layers):
        layers = []
        n_in = encoder_layers[0]
        for n_out in encoder_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def build_decoder(self, latent_dim, decoder_layers):
        layers = []
        n_in = latent_dim
        for n_out in decoder_layers:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def _build_mean_var(self, encoder_layers, latent_dim):
        return nn.Linear(encoder_layers[-1], latent_dim)
    
    build_mean = _build_mean_var
    build_var = _build_mean_var