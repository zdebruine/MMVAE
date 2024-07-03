import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Union, Literal, Optional, Callable

from ._fc_block import FCBlock

def _identity(x):
    return x
            

class Encoder(nn.Module):
    """
    Encoder module for a Variational Autoencoder (VAE) with flexible configurations.

    Args:
        n_in (int): Number of input features.
        n_hidden (int): Number of hidden units in the encoder.
        n_out (int): Number of output features.
        fc_layers (list[int], optional): List specifying the number of units in each fully connected layer. Defaults to an empty list.
        distribution (Union[Literal['ln'], Literal['normal']], optional): Type of distribution for the latent variables. Defaults to 'normal'.
        var_activation (Optional[Callable], optional): Activation function for the variance. Defaults to torch.exp.
        return_dist (bool, optional): Whether to return the distribution object. Defaults to False.
        var_eps (float, optional): Small epsilon value for numerical stability in variance calculation. Defaults to 1e-4.
        **fc_block_kwargs: Additional keyword arguments for the fully connected block.

    Attributes:
        encoder (FCBlock): Fully connected block for encoding input features.
        mean_encoder (nn.Linear): Linear layer to compute the mean of the latent variables.
        var_encoder (nn.Linear): Linear layer to compute the variance of the latent variables.
        z_transformation (Callable): Transformation applied to the latent variables. Defaults to softmax for log-normal distribution.
        var_activation (Callable): Activation function for the variance. Defaults to torch.exp.
        var_eps (float): Small epsilon value for numerical stability.
        return_dist (bool): Whether to return the distribution object.
    """
    
    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: int,
        fc_layers: list[int] = [],
        distribution: Union[Literal['ln'], Literal['normal']] = 'normal',
        var_activation: Optional[Callable] = torch.exp,
        return_dist: bool = False,
        var_eps: float = 1e-4, # numerical stability
        **fc_block_kwargs
    ):
        super().__init__()
        
        # Fully connected block for encoding the input features
        self.encoder = FCBlock([n_in, *fc_layers, n_hidden], **fc_block_kwargs)
        
        # Linear layer to compute the mean of the latent variables
        self.mean_encoder = nn.Linear(n_hidden, n_out)
        
        # Linear layer to compute the variance of the latent variables
        self.var_encoder = nn.Linear(n_hidden, n_out)
        
        # Transformation for latent variables (softmax for log-normal distribution, identity otherwise)
        self.z_transformation = nn.Softmax(dim=-1) if distribution == "ln" else _identity
        
        # Activation function for the variance
        self.var_activation = var_activation if callable(var_activation) else _identity
        
        # Small epsilon value for numerical stability
        self.var_eps = var_eps
        
        # Whether to return the distribution object
        self.return_dist = return_dist
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[Tuple[torch.distributions.Normal, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: 
                If return_dist is True, returns the distribution and latent variables.
                Otherwise, returns the mean, variance, and latent variables.
        """
        # Encode the input features
        q = self.encoder(x)
        
        # Compute the mean of the latent variables
        q_m = self.mean_encoder(q)
        
        # Compute the variance of the latent variables and add epsilon for numerical stability
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        
        # Create a normal distribution with the computed mean and variance
        dist = Normal(q_m, q_v.sqrt())
        
        # Sample the latent variables and apply the transformation
        latent = self.z_transformation(dist.rsample())
        
        if self.return_dist:
            return dist, latent
        
        return q_m, q_v, latent
