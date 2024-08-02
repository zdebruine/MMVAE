from typing import Literal, Optional, Type, Union
from collections import OrderedDict
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Normal


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
        layers: list[int],
        dropout_rate: list[float] = 0.0,
        use_batch_norm: list[bool] = False,
        use_layer_norm: list[bool] = False,
        activation_fn: Optional[Literal['ReLU']] = None,
    ):
        super().__init__()
        # Duplicate layers if only one found for n_in and n_out
        if len(layers) == 1:
            layers = layers * 2 
        # Pair layers into n_in n_out pairs in sequence
        in_outs = list(zip(layers[0:], layers[1:]))
        self.fc_layers = nn.Sequential(
            OrderedDict([(
                str(i), nn.Sequential(*(
                    layer for layer in (
                        nn.Linear(n_in, n_out),
                        nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm[i] else None,
                        nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm[i] else None,
                        activation_fn[i]() if activation_fn[i] else None,
                        nn.Dropout(p=dropout_rate[i]) if dropout_rate[i] > 0 else None,
                    ) if layer is not None)
                ))
                for i, (n_in, n_out) in enumerate(in_outs)
            ])
        )
        
        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the fully connected block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc_layers(x)
    

class FCBlockConfig:
    
    def __init__(
        self,
        layers: list[int],
        dropout_rate: Union[float, list[float]] = 0.0,
        use_batch_norm: Union[bool, list[bool]] = False,
        use_layer_norm: Union[bool, list[bool]] = False,
        activation_fn: Union[Optional[Type[nn.Module]], list[Optional[Type[nn.Module]]]] = None
    ):
        super().__init__()
        self._validate_and_set_layers(layers)
        self._validate_and_set_option('dropout_rate', dropout_rate, float)
        self._validate_and_set_option('use_batch_norm', use_batch_norm, bool)
        self._validate_and_set_option('use_layer_norm', use_layer_norm, bool, )
        self._validate_and_set_option('activation_fn', activation_fn, nn.Module, comparison_fn=issubclass, optional=True)
    
    def _validate_and_set_layers(self, layers):
        # Assert layers is a list
        assert isinstance(layers, list), f"layers must be a list found type: {type(layers)}"
        # Assert all values are integer objects greater than 0
        assert all(isinstance(layer, int) and layer > 0 for layer in layers), "layers must be positive integers"
        n_layers = len(layers)
        # Since layers will be paired into n_in and n_out
        # the number of layers will be equal to 1 for both 
        # length of 1 and 2, otherwise the number of layers 
        # will then be one less len(layers) because pairing
        # starting from first two elements
        if n_layers > 1:
            n_layers -= 1
        self.n_layers = n_layers
        self.layers = layers
        
    def _validate_and_set_option(self, name, obj, req_type, comparison_fn=isinstance, optional=False):
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
            value = obj
        elif obj == None or comparison_fn(obj, req_type):
            value = [obj] * self.n_layers
        else:
            raise ValueError(f"{name} ({obj}) is not a {str(req_type)} or iterable of {req_type}")
        
        setattr(self, name, value)
        
class FCBlock(BaseFCBlock):
    
    def __init__(self, config: FCBlockConfig):
        super(FCBlock, self).__init__(
            layers=config.layers,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            activation_fn=config.activation_fn
        )


class ConditionalLayer(nn.Module):
    
    def __init__(self, batch_key: str, conditions_path: str, fc_block_config: FCBlockConfig):
        super(ConditionalLayer, self).__init__()
        
        self.batch_key = batch_key
        self.unique_conditions = { self.format_condition_key(condition) for condition in pd.read_csv(conditions_path, header=None)[0] }
        
        self.conditions = nn.ModuleDict({
            condition: FCBlock(fc_block_config) 
            for condition in self.unique_conditions
        })
        
    def format_condition_key(self, condition: str):
        return condition.replace('.', '_')
    
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, condition: Optional[str] = None):
        
        if condition:
            return self.conditions[condition](x), metadata
        
        active_conditions = set()
        xhat = torch.zeros_like(x)
        
        for condition, group_df in metadata.groupby(self.batch_key):
            condition = self.format_condition_key(condition)
            mask = torch.zeros(len(metadata), dtype=int, device=x.device)
            mask[group_df.index] = 1
            xhat = xhat + self.conditions[condition](x * mask.unsqueeze(1))
            active_conditions.add(condition)
            
        self.active_condition_modules = active_conditions
        
        return xhat
    
    
class ConditionalLayers(nn.Module):
    
    def __init__(
        self,
        conditional_paths: dict[str, str],
        fc_block_config: FCBlockConfig,
        selection_order: Optional[list[str]] = None,
    ):
        super(ConditionalLayers, self).__init__()
        
        if not selection_order:
            selection_order = list(conditional_paths.keys())
            self.shuffle_selection_order = True
        else:
            self.shuffle_selection_order = False
            
        self.selection_order = torch.arange(0, len(selection_order), dtype=torch.int32, requires_grad=False)
        
        self.layers: list[ConditionalLayer] = nn.ModuleList([
            ConditionalLayer(batch_key, conditional_paths[batch_key], fc_block_config)
            for batch_key in selection_order
        ])
        
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, conditions: Optional[dict[str, str]] = None):
        order = self.selection_order
        
        if self.shuffle_selection_order:
            permutation = torch.randperm(self.selection_order.size(0))
            order = order[permutation]
            
        for idx in order:
            x = self.layers[idx](x, metadata)
        
        return x
    



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
        latent_dim: int,
        fc_block_config: FCBlockConfig,
        distribution: Union[Literal['ln'], Literal['normal']] = 'normal',
        return_dist: bool = False,
        var_eps: float = 1e-4, # numerical stability
    ):
        super().__init__()

        # Fully connected block for encoding the input features
        self.encoder = FCBlock(fc_block_config)
        
        # Get hidden and latent dimenision from layer dim list
        n_hidden = fc_block_config.layers[-1]
        
        # Linear layer to compute the mean of the latent variables
        self.mean_encoder = nn.Linear(n_hidden, latent_dim)
        
        # Linear layer to compute the variance of the latent variables
        self.var_encoder = nn.Linear(n_hidden, latent_dim)
        
        # Transformation for latent variables (softmax for log-normal distribution, identity otherwise)
        self.z_transformation = nn.Softmax(dim=-1) if distribution == "ln" else _identity
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
        q_v = torch.exp(self.var_encoder(q)) + self.var_eps
        
        # Create a normal distribution with the computed mean and variance
        dist = Normal(q_m, q_v.sqrt())
        
        # Sample the latent variables and apply the transformation
        latent = self.z_transformation(dist.rsample())
        
        if self.return_dist:
            return dist, latent
        
        return q_m, q_v, latent




class BaseExpert(nn.Module):
    
    def __init__(
        self,
        id: str,
        encoder: FCBlock,
        decoder: FCBlock
    ):
        super().__init__()
        
        self.id = id
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x: torch.Tensor):
        return self.encoder(x)
        
    def decode(self, x: torch.Tensor):
        return self.decoder(x)
    
class Expert(BaseExpert):
    
    def __init__(
        self,
        id: str,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig, 
    ):
        super(Expert, self).__init__(
            id=id, 
            encoder=FCBlock(encoder_config), 
            decoder=FCBlock(decoder_config)
        )
    

class Experts(nn.ModuleDict):
    
    def __init__(self, experts: list[BaseExpert]):
        super().__init__({ expert.id: expert for expert in experts})
        

