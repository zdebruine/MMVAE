from typing import Callable, Literal, NamedTuple, Optional, Union, Iterable

import torch
from torch.distributions import Normal
from .mixins.init import HeWeightInitMixIn

from sciml.utils.constants import REGISTRY_KEYS as RK
from .base import BaseModule, Encoder, FCBlock
from ._vae import VAE

class SimpleVAE(VAE, HeWeightInitMixIn, BaseModule):
    """
    A simple Variational Autoencoder (VAE) class.

    This class extends VAE, HeWeightInitMixIn, and BaseModule, and provides a basic implementation of a VAE with optional weight initialization.
    
    Args:
        init_weights (bool, optional): Whether to initialize weights using He initialization. Defaults to True.
        *args: Additional positional arguments for the parent classes.
        **kwargs: Additional keyword arguments for the parent classes.
    """
    
    def __init__(self, init_weights: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if init_weights:
            self.init_weights()  # Initialize weights using He initialization
    
    def get_module_inputs(self, batch, **module_input_kwargs):
        """
        Prepare the inputs for the module.

        Args:
            batch (dict): A batch of data.
            **module_input_kwargs: Additional keyword arguments.

        Returns:
            tuple: Arguments and keyword arguments for the forward pass.

        Raises:
            ValueError: If the batch does not contain the expected key 'RK.X'.
            TypeError: If the data type for 'batch[RK.X]' is invalid.
        """
        # Check if the batch contains tensors and convert if necessary
        if isinstance(batch, dict) and RK.X in batch:
            if isinstance(batch[RK.X], torch.Tensor):
                args = (batch[RK.X],)
            else:
                try:
                    args = (torch.tensor(batch[RK.X]),)
                except ValueError:
                    raise TypeError(f"Invalid data type for batch[RK.X]: {type(batch[RK.X])}")
        else:
            raise ValueError(f"Batch does not contain the expected key '{RK.X}' - ({batch.keys()})")
        
        return args, module_input_kwargs
    
    def configure_optimizers(self):
        """
        Configure optimizers for the VAE.

        Returns:
            torch.optim.Adam: Optimizer for the encoder and decoder parameters.
        """
        return torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 1e-3, "weight_decay": 1e-8},
            {'params': self.decoder.parameters(), 'lr': 1e-3,  "weight_decay": 1e-8},
        ])
    

