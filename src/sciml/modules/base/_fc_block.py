from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import OrderedDict
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
    except TypeError:
        return False
    return True
    

class BaseFCBlock(nn.Module):
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
    
    def __init__(
        self,
        layers: Iterable[int],
        dropout_rate: Union[float, Iterable[float]] = 0.0,
        use_batch_norm: Union[bool, Iterable[bool]] = False,
        use_layer_norm: Union[bool, Iterable[bool]] = False,
        activation_fn: Union[Optional[nn.Module], Iterable[Optional[nn.Module]]] = None,
    ):
        super().__init__()
        
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise ValueError("All elements in 'layers' must be positive integers")

        self.n_layers = len(layers)

        dropout_rate = self._validate_and_get(dropout_rate, float)
        use_batch_norm = self._validate_and_get(use_batch_norm, bool)
        use_layer_norm = self._validate_and_get(use_layer_norm, bool)
        activation_fn = self._validate_and_get(activation_fn, nn.Module, comparison_fn=issubclass, optional=True)
        
        # Construct the fully connected layers
        self.fc_layers = nn.Sequential(
            OrderedDict([(
                f"{i}",
                nn.Sequential(*(
                    layer for layer in (
                        nn.Linear(n_in, n_out),
                        nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm[i] else None,
                        nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm[i] else None,
                        activation_fn[i]() if activation_fn[i] else None,
                        nn.Dropout(p=dropout_rate[i]) if dropout_rate[i] > 0 else None,
                    ) if layer is not None)
                ))
                for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:]))
            ])
        )
        
        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        
    def _validate_and_get(self, obj, name, req_type, comparison_fn=isinstance, optional=False):
        types = (req_type, type(None)) if optional else (req_type,)

        if not optional and obj == None:
            raise ValueError(f"{name} is not optional but value is None")
        
        if is_iterable(obj):
            if len(obj) != self.n_layers:
                raise ValueError(f"Length of '{name}' must match the length of 'layers': {len(obj)} != {self.n_layers}")
            try:
                assert all(val != None and comparison_fn(val, types) for val in obj if not (optional and val == None))
            except (AssertionError, TypeError):
                raise ValueError(f"All elements in '{name}' must be a {str(req_type)}")
            return obj
        elif obj == None or comparison_fn(obj, req_type):
            return [obj] * self.n_layers
        else:
            raise ValueError(f"{name} ({obj}) is not a {str(req_type)} or iterable of {req_type}")
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the fully connected block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc_layers(x)
    
    
@dataclass
class FCBlockConfig:
    
    layers: Iterable[int]
    dropout_rate: Union[float, Iterable[float]] = 0.0
    use_batch_norm: Union[bool, Iterable[bool]] = False
    use_layer_norm: Union[bool, Iterable[bool]] = False
    activation_fn: Union[Optional[nn.Module], Iterable[Optional[nn.Module]]] = None
    
    
class FCBlock(BaseFCBlock):
    
    def __init__(self, config: FCBlockConfig):
        super(FCBlock, self).__init__(
            layers=config.layers,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            activation_fn=config.activation_fn
        )