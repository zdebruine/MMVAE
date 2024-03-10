from typing import Tuple, Union, Any
import torch
import torch.nn as nn
import mmvae.models.utils as utils

class Expert(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator = None):
        super(Expert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        raise NotImplementedError()

class VAE(nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x: torch.Tensor) -> Tuple[Union[Any, torch.Tensor], Any, Any, Tuple[Any]]:
        
        x = self.encoder(x)
        mu = self.mean(x) 
        log_var = self.var(x)
        
        x = self.reparameterize(mu, log_var)
        x = self.decoder(x)
        return x, mu, log_var