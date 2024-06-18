import torch
import torch.nn as nn

from sciml.utils.constants import REGISTRY_KEYS as RK, ModelOutputs

class VAEMixIn:
    """
    Defines vae forward pass.
    Expectes encoder, decoder, fc_mean, fc_var to be defined
    """
    
    encoder: nn.Module
    decoder: nn.Module
    fc_mean: nn.Module
    fc_var: nn.Module
    
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
    
    def forward(self, batch_dict):
        x = batch_dict.get(RK.X)
        metadata = batch_dict.get(RK.METADATA)
        
        qzm, qzv = self.encode(x)
        z = self.reparameterize(qzm, qzv)
        z_star = self.after_reparameterize(z, metadata)
        x_hat = self.decode(z_star)
        
        return ModelOutputs(
            qzm = qzm,
            qzv = qzv,
            z = z, 
            z_star = z_star,
            x_hat = x_hat, 
        )