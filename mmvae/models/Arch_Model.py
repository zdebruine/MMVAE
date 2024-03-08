import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60664, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 256),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 60664),
            nn.LeakyReLU()
    
        )

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
    
