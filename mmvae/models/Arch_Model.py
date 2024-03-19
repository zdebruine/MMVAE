import torch
import torch.nn as nn
import mmvae.models.utils as utils

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60664, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 0.8),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768, 0.8),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256, 0.8)
        )
        
        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768, 0.8),
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 0.8),
            nn.Linear(1024, 60664),
            nn.Sigmoid(),
        )

        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)
        utils._submodules_init_weights_xavier_uniform_(self.fc_mu)
        utils._xavier_uniform_(self.fc_var, -1.0)

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
