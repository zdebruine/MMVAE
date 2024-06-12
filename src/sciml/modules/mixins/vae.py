import torch

from sciml.utils.constants import REGISTRY_KEYS as RK

class VAEMixIn:
    """
    Defines vae forward pass.
    Expectes encoder, decoder, fc_mean, fc_var to be defined
    """
    
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