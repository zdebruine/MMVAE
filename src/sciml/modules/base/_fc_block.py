import torch
import torch.nn as nn

import collections
from typing import Iterable, Optional, Union

def is_iterable(obj):
    """
    Check if an object is iterable.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is iterable, False otherwise.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False

class FCBlock(nn.Module):
    """
    Fully Connected Block (FCBlock) for building neural network layers.

    This class constructs a series of fully connected layers with optional dropout, batch normalization,
    layer normalization, and activation functions.

    Args:
        layers (Iterable[int]): A sequence of integers specifying the number of units in each layer.
        dropout_rate (Union[float, Iterable[float]], optional): Dropout rate(s) for each layer. Defaults to 0.0.
        use_batch_norm (Union[bool, Iterable[bool]], optional): Whether to use batch normalization for each layer. Defaults to False.
        use_layer_norm (Union[bool, Iterable[bool]], optional): Whether to use layer normalization for each layer. Defaults to False.
        activation_fn (Union[Optional[nn.Module], Iterable[Optional[nn.Module]]], optional): Activation function(s) for each layer. Defaults to nn.ReLU.

    Attributes:
        fc_layers (nn.Sequential): The sequential container of fully connected layers.
    """
    
    __initialized = False
    
    def __init__(
        self,
        layers: Iterable[int], 
        dropout_rate: Union[float, Iterable[float]] = 0.0,
        use_batch_norm: Union[bool, Iterable[bool]] = False,
        use_layer_norm: Union[bool, Iterable[bool]] = False,
        activation_fn: Union[Optional[nn.Module], Iterable[Optional[nn.Module]]] = nn.ReLU,
    ):  
        super().__init__()
        
        self.layers = layers
        
        # Validate and mask the dropout_rate, use_batch_norm, use_layer_norm, and activation_fn
        dropout_rate = self._validate_and_mask(dropout_rate, float)
         
        try:
            assert all(dr >= 0 for dr in dropout_rate)
        except AssertionError:
            raise ValueError("Dropout rate must be greater than or equal to 0")
        
        use_batch_norm = self._validate_and_mask(use_batch_norm, bool)
        use_layer_norm = self._validate_and_mask(use_layer_norm, bool)
        activation_fn = self._validate_and_mask(activation_fn, nn.Module, type_comparison_fn=issubclass)
            
        # Construct the fully connected layers
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            *[
                                layer for layer in [
                                    nn.Linear(n_in, n_out),
                                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm[i] else None,
                                    nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm[i] else None,
                                    activation_fn[i]() if activation_fn[i] else None,
                                    nn.Dropout(p=dropout_rate[i]) if dropout_rate[i] > 0 else None,
                                ] if layer is not None
                            ]
                        )
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
                ]
            )
        )
        
        self.__initialized = True
        
    def _validate_and_mask(self, kwargs, cls, type_comparison_fn = isinstance):
        """
        Validate and mask keyword arguments for consistency with the number of layers.

        Args:
            kwargs: The keyword arguments to validate and mask.
            _type: The expected type of the keyword arguments.
            optional (bool, optional): Whether the keyword arguments are optional. Defaults to False.
            default (optional): The default value if kwargs is not provided. Defaults to None.

        Returns:
            list: A list of validated and masked keyword arguments.
        """
        if not is_iterable(kwargs):
            kwargs = [kwargs] * len(self.layers)
            
        try:
            assert len(kwargs) == len(self.layers)
        except AssertionError:
            raise ValueError(f"Iterable kwarg not equal size of layers! kwargs: {len(kwargs)} layers: {len(self.layers)}")
        
        try:
            assert all(type_comparison_fn(val, cls) for val in kwargs if val is not None)
        except AssertionError:
            raise ValueError("All values in kwarg must be of the same type")
            
        return kwargs
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the fully connected block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc_layers(x)
    
    @property
    def layers(self) -> tuple[int]:
        """
        Get the layers of the fully connected block.

        Returns:
            tuple[int]: The layers of the fully connected block.
        """
        return self._layers
    
    @layers.setter
    def layers(self, value: Iterable[int]):
        """
        Set the layers of the fully connected block.

        Args:
            value (Iterable[int]): The layers to set.

        Raises:
            RuntimeError: If attempting to set layers after initialization.
        """
        if self.__initialized:
            raise RuntimeError("Cannot set layers after initialization")
        self._layers = tuple(value)
    
    @property
    def input_dim(self):
        """
        Get the input dimension of the fully connected block.

        Returns:
            int: The input dimension.
        """
        return self.layers[0]
    
    @property
    def output_dim(self):
        """
        Get the output dimension of the fully connected block.

        Returns:
            int: The output dimension.
        """
        return self.layers[-1]
