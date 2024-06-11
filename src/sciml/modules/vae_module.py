import torch
import torch.nn as nn
from sciml.utils.constants import REGISTRY_KEYS as RK
        
        

class VAEMixIn:
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.before_reparameterize(x)
        return self.fc_mean(x), self.fc_var(x)
        
    def before_reparameterize(self, x):
        return x
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=std.device)
        return mu + eps * std
    
    def after_reparameterize(self, z, metadata):
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, input_dict):
        
        x = input_dict[RK.X]
        metadata = input_dict.get(RK.METADATA)
        
        qzm, qzv = self.encode(x)
        z = self.reparameterize(qzm, qzv)
        z_star = self.after_reparameterize(z, metadata)
        x_hat = self.decode(z_star)
        
        return {
            RK.QZM: qzm,
            RK.QZV: qzv,
            RK.Z: z, 
            RK.Z_STAR: z_star,
            RK.X_HAT: x_hat, 
        }
        
class BasicVAE(VAEMixIn, nn.Module):
    
    def __init__(
        self,
        encoder_layers = [60664, 1024, 512], 
        latent_dim = 256, 
        decoder_layers = [512, 1024, 60664],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.build()
        
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
    
    def build(self):
        """Returns dictinoary of encoder, decoder, mean, var keys and their instaniated builds"""
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.fc_mean = self.build_mean()
        self.fc_var = self.build_var()