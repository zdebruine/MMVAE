import torch
import torch.nn as nn

class Expert(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module):
        super(Expert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        """
        Forward pass treated as autoencoder
        >>> return self.decoder(self.encoder(x))
        """
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.discriminator(x)
        return x

class VAE(nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: nn.ModuleList, decoder: nn.ModuleList, mean: nn.Module, var: nn.Module) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        
    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x, encoder_outputs = self.encoder(x)
        mu = self.mean(x)
        var = self.var(x)
        x = self.reparameterize(mu, torch.exp(0.5 * var))
        x, decoder_outputs = self.decoder(x)
        return x, mu, var, encoder_outputs, decoder_outputs