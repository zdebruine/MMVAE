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

class MultiClassDiscriminator(nn.Module):
    """
    The Multi-Class Discriminator class is designed to be used by the MMVAE to discourage bias
    for any one expert in the the shared latent space.

    Args:
        layers: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a torch.Tensor
                The output size should match the number of classes, and the output must be raw prediction scores
                PyTorch's CrossEntropy function expects unnormalized logits as input

        classes: List of the expert names, used in getting the proper label that will then be passed to the loss function

    Vars:
        labels: NxN "Identity Matrix" (torch.Tensor) that corresponds to each classes expected output | N = len(classes) 
    """
    def __init__(self, layers: nn.Sequential, classes: list[str]) -> None:
        super(MultiClassDiscriminator, self).__init__()
        self.layers = layers
        self.labels = torch.eye(len(classes))
        self.classes = {}
        # Classes maps the expert names to an index value to index the labels
        for i, label in enumerate(classes):
            self.classes.update({label:i})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def get_label(self, key: str) -> torch.Tensor:
        """
        Retrive the expected/target prediction based on the current expert
        """
        return self.labels[self.classes[key]]

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