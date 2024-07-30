import torch
import torch.nn as nn
from typing import Iterable, Union, Optional, Any
from ._fc_block import FCBlock
from sciml.utils.constants import REGISTRY_KEYS as RK

def parse_kwargs(kwargs: dict[str, Any], target: str):
    """
    Parse keyword arguments to extract those containing a specific target string.

    Args:
        kwargs (dict[str, Any]): Dictionary of keyword arguments.
        target (str): Target string to search for in the keys.

    Returns:
        dict: Filtered dictionary with keys containing the target string.
    """
    return {kwarg.rstrip(target) for kwarg in kwargs if target in kwarg}

class Expert(nn.ModuleDict):
    """
    A module dictionary for managing encoder and decoder components of an expert.

    Args:
        encoder (FCBlock): The encoder module.
        decoder (FCBlock): The decoder module.

    Attributes:
        ENCODER (str): Key for the encoder module.
        DECODER (str): Key for the decoder module.
    """
    
    ENCODER = 'encoder'
    DECODER = 'decoder'
    
    def __init__(
        self,
        encoder: FCBlock,
        decoder: FCBlock
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def encoder(self):
        """
        Get the encoder module.

        Returns:
            FCBlock: The encoder module.
        """
        return self[self.ENCODER]
    
    @encoder.setter
    def encoder(self, module):
        """
        Set the encoder module.

        Args:
            module (FCBlock): The encoder module.

        Raises:
            RuntimeError: If attempting to override an existing encoder module.
        """
        if self.ENCODER in self:
            raise RuntimeError("Attempting to override encoder!")
        self.add_module(self.ENCODER, module)
        
    @property
    def decoder(self):
        """
        Get the decoder module.

        Returns:
            FCBlock: The decoder module.
        """
        return self[self.DECODER]
    
    @decoder.setter
    def decoder(self, module):
        """
        Set the decoder module.

        Args:
            module (FCBlock): The decoder module.

        Raises:
            RuntimeError: If attempting to override an existing decoder module.
        """
        if self.DECODER in self:
            raise RuntimeError("Attempting to override decoder!")
        self.add_module(self.DECODER, module)
    
    def encode(self, x: torch.Tensor):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded output tensor.
        """
        return self.encoder(x)
        
    def decode(self, x: torch.Tensor):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Decoded output tensor.
        """
        return self.decoder(x)

class Experts(nn.ModuleDict):
    """
    A module dictionary for managing human and mouse experts.

    Args:
        human_encoder_kwargs (dict[str, Any]): Keyword arguments for the human encoder.
        human_decoder_kwargs (dict[str, Any]): Keyword arguments for the human decoder.
        mouse_encoder_kwargs (dict[str, Any]): Keyword arguments for the mouse encoder.
        mouse_decoder_kwargs (dict[str, Any]): Keyword arguments for the mouse decoder.
    """
    
    def __init__(
        self, 
        human_encoder_kwargs: dict[str, Any],
        human_decoder_kwargs: dict[str, Any],
        mouse_encoder_kwargs: dict[str, Any],
        mouse_decoder_kwargs: dict[str, Any],
    ):
        super().__init__()
        
        # Initialize human expert
        self.human = Expert(
            encoder=FCBlock(**human_encoder_kwargs),
            decoder=FCBlock(**human_decoder_kwargs),
        )
        
        # Initialize mouse expert
        self.mouse = Expert(
            encoder=FCBlock(**mouse_encoder_kwargs),
            decoder=FCBlock(**mouse_decoder_kwargs),
        )
        
    @property
    def mouse(self):
        """
        Get the mouse expert.

        Returns:
            Expert: The mouse expert module.
        """
        return self[RK.MOUSE]
    
    @mouse.setter
    def mouse(self, module):
        """
        Set the mouse expert.

        Args:
            module (Expert): The mouse expert module.

        Raises:
            RuntimeError: If attempting to override an existing mouse expert module.
        """
        if RK.MOUSE in self:
            raise RuntimeError("Attempting to override mouse expert module. Exception Raised!") 
        self.add_module(RK.MOUSE, module)

    @property
    def human(self):
        """
        Get the human expert.

        Returns:
            Expert: The human expert module.
        """
        return self[RK.HUMAN]
    
    @human.setter
    def human(self, module):
        """
        Set the human expert.

        Args:
            module (Expert): The human expert module.

        Raises:
            RuntimeError: If attempting to override an existing human expert module.
        """
        if RK.HUMAN in self:
            raise RuntimeError("Attempting to override human expert module. Exception Raised!") 
        self.add_module(RK.HUMAN, module)
