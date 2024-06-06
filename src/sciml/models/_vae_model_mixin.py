import torch
from sciml._constant import REGISTRY_KEYS as RK



class VAEModelMixIn:
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.before_reparameterize(x)
        return self.fc_mean(x), self.fc_var(x)
        
    def before_reparameterize(self, x):
        return x
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps * std
    
    def after_reparameterize(self, z, metadata):
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, metadata):
        
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