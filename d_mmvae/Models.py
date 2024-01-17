import torch
import torch.nn as nn
from torch import Tensor

class MultiClassDiscriminator(nn.Module):
    """
    The Multi-Class Discriminator class is designed to be used by the MMVAE to discourage bias
    for any one expert in the the shared latent space.

    Args:
        layers: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a Tensor
                The output size should match the number of classes, and the output must be raw prediction scores
                PyTorch's CrossEntropy function expects unnormalized logits as input

        classes: List of the expert names, used in getting the proper label that will then be passed to the loss function

    Vars:
        labels: NxN "Identity Matrix" (tensor) that corresponds to each classes expected output | N = len(classes) 
    """
    def __init__(self, layers: nn.Sequential, classes: list[str]) -> None:
        super(MultiClassDiscriminator, self).__init__()
        self.layers = layers
        self.labels = torch.eye(len(classes))
        self.classes = {}
        # Classes maps the expert names to an index value to index the labels
        for i, label in enumerate(classes):
            self.classes.update({label:i})

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
    
    def get_label(self, key: str) -> Tensor:
        """
        Retrive the expected/target prediction based on the current expert
        """
        return self.labels[self.classes[key]]
    

class Expert:
    """
    The Expert class is designed to be used by the MMVAE to handle the non-shared portions
    of the overall network. The Encoder, Decoder, and expert-specific Discriminator.

    Args:
        encoder: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a Tensor
                 The first stage of the MMVAE forward pass and is expert-specific, the output size of the encoder should 
                 match the input size of the shared space and match across all experts

        decoder: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a Tensor
                 The last stage of the MMVAE forward pass and is expert-specific, the input size of the decoder should 
                 match the output size of the shared space and match across all experts

        discriminator: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a Tensor
                       Should output a binary classification on if the data is real or fake

        expert_name: The name of the expert/modality
    """
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, discriminator: nn.Sequential, expert_name: str) -> None:
        super(Expert, self).__init__()
        self.encoder = encoder
        self.decorder = decoder
        self.discriminator = discriminator
        self.label = expert_name

    def encoder(self):
        if isinstance(self.encoder, nn.Sequential, None) is None:
            raise ValueError("Encoder is not a nn.Sequential")
        return self.encoder
    
    def __str__(self) -> str:
        return self.label

class MMVAE(nn.Module):
    """
    The MMVAE class is designed to implement a shared VAE that is fed input from multiple-modalities.

    Args:
        encoder: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a Tensor
                 Encoder portion of the shared space. The input size should match the output of the expert-specific encoder
                 networks. The output of the encoder should match the input size of the Multi-Class Discriminator and the
                 input size of the learned distributions in the latent space.

        decoder: Designed for an nn.Sequential stack of layers, but can be any object that is callable and returns a Tensor
                 Decoder portion of the shared space. The output size should match the input of the expert-specific decoder
                 networks. The input of the decoder should match the output size of the learned distributions in the latent space.

        mean: Single nn.Linear layer that represents the mean of the learned distribution in the latent space. Should be of size
              NxM where N is the output size of the Encoder and M is the latent dimension

        var: Single nn.Linear layer that represents the variance of the learned distribution in the latent space. Should be of size
              NxM where N is the output size of the Encoder and M is the latent dimension
        
        experts: List of Expert objects, one for each individual modality/expert  
            
        discriminator: Instance of a Multi-Class Discriminator

    Vars:
        current_expert: Int representing the index of the current expert that is being trained. Control is designed to be handled
                        outside of the Model class in the Trainer class instead as that pairs with the dataloading for each expert
    """
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, mean: nn.Linear,
                 var: nn.Linear, experts: list[Expert], discriminator: MultiClassDiscriminator) -> None:
        super(MMVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        self.experts = experts
        self.discriminator = discriminator
        self.current_expert = 0

    def reparameterize(self, mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps

    def forward(self, x: Tensor) -> tuple[Tensor]:
        """
        Implements the complex forward pass of the network. Feeds through the expert-specific encoder,
        the shared space encoder, the multi-class discriminator and shared latent space, the shared space
        decoder, the expert-specific decoder, and finally the expert-specific discriminator.

        Returns:
            x: Generated output of the model for the given expert

            mu: Mean vector of the latent distribution

            var: Variance vector of the latent distribution

            adv_feedback: Mulit-Class adversarial feedback from the encoder portion

            exp_feedback: Expert discriminator output on generated data
        """
        x = self.experts[self.current_expert].encoder(x)
        x = self.encoder(x)
        adv_feedback = self.discriminator(x)
        mu = self.mean(x)
        var = self.var(x)
        x = self.reparameterize(mu, torch.exp(0.5 * var))
        x = self.decoder(x)
        x = self.experts[self.current_expert].forward_decode(x)
        exp_feedback = self.experts[self.current_expert].get_adversarial_feedback(x)
        return x, mu, var, adv_feedback, exp_feedback
    
    def set_current_expert(self, new_expert: int) -> None:
        self.current_expert = new_expert

class VAE(nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential, mean: nn.Linear, var: nn.Linear,) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        
    def reparameterize(self, mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps

    def forward(self, x: Tensor) -> tuple[Tensor]:
        x = self.encoder(x)
        mu = self.mean(x)
        var = self.var(x)
        z = self.reparameterize(mu, torch.exp(0.5 * var))
        x_hat = self.decoder(z)
        return x_hat, mu, var
