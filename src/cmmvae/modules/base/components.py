from typing import List, Literal, Optional, Type, TypeVar, Union
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

T = TypeVar('T')


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
        layers (List[int]): A sequence of integers specifying the number of units in each layer.
        dropout_rate (Union[float, List[float]], optional): Dropout rate(s) for each layer. Defaults to 0.0.
        use_batch_norm (Union[bool, List[bool]], optional): Whether to use batch normalization for each layer. Defaults to False.
        use_layer_norm (Union[bool, List[bool]], optional): Whether to use layer normalization for each layer. Defaults to False.
        activation_fn (Union[Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]], optional): Activation function(s) for each layer. Defaults to nn.ReLU.
        return_hidden (Union[bool, List[bool]], optional): Whether to aggregate and return hidden representations. Defaults to False.

    Attributes:
        fc_layers (nn.Sequential): The sequential container of fully connected layers.
        n_layers (int): Number of layers in the network.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
    """

    def __init__(
        self,
        layers: List[int],
        dropout_rate: Union[float, List[float]] = 0.0,
        use_batch_norm: Union[bool, List[bool]] = False,
        use_layer_norm: Union[bool, List[bool]] = False,
        activation_fn: Union[Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]] = nn.ReLU,
        return_hidden: Union[bool, List[bool]] = False
    ):
        super().__init__()

        # Duplicate layers if only one found for n_in and n_out
        if len(layers) == 1:
            layers = layers * 2

        # Pair layers into n_in, n_out pairs in sequence
        layers = list(zip(layers[:-1], layers[1:]))
        self.n_layers = len(layers)

        # Ensure input arguments are lists with correct length
        dropout_rate = self.to_list(dropout_rate)
        use_batch_norm = self.to_list(use_batch_norm)
        use_layer_norm = self.to_list(use_layer_norm)
        activation_fn = self.to_list(activation_fn)
        self.return_hidden = self.to_list(return_hidden)

        # Create fully connected layers
        fc_layers = [
            self._make_layer(n_in, n_out, use_batch_norm[i], use_layer_norm[i], activation_fn[i], dropout_rate[i])
            for i, (n_in, n_out) in enumerate(layers)
        ]

        self.fc_layers = nn.Sequential(*fc_layers)

        self.input_dim = layers[0][0]
        self.output_dim = layers[-1][1]

    @property
    def can_bypass(self) -> bool:
        """
        Check if the model can bypass the hidden layer return logic.

        Returns:
            bool: True if no hidden layers are returned, False otherwise.
        """
        return not any(self.return_hidden)

    def _make_layer(self, n_in: int, n_out: int, use_batch_norm: bool, use_layer_norm: bool, activation_fn: Optional[Type[nn.Module]], dropout_rate: float) -> nn.Sequential:
        """
        Create a single fully connected layer.

        Args:
            n_in (int): Number of input units.
            n_out (int): Number of output units.
            use_batch_norm (bool): Whether to use batch normalization.
            use_layer_norm (bool): Whether to use layer normalization.
            activation_fn (Optional[Type[nn.Module]]): Activation function.
            dropout_rate (float): Dropout rate.

        Returns:
            nn.Sequential: A sequence of layers comprising the fully connected layer.
        """
        layers = [nn.Linear(n_in, n_out)]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001))
        if use_layer_norm:
            layers.append(nn.LayerNorm(n_out, elementwise_affine=False))
        if activation_fn is not None:
            layers.append(activation_fn(dim=1) if issubclass(activation_fn, nn.Softmax) else activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """
        Forward pass through the fully connected block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]: Output tensor or tuple of output and hidden representations if required.
        """
        if self.can_bypass:
            return self.fc_layers(x)

        hidden_representations = []
        for i, layer in enumerate(self.fc_layers):
            for name, sublayer in layer.named_children():
                x = sublayer(x)
                if name == '2' and self.return_hidden[i]:
                    hidden_representations.append(x)
        return x, hidden_representations

    def to_list(self, value: Union[T, List[T]]) -> List[T]:
        """
        Ensure that a value is a list with the correct number of elements.

        Args:
            value (Union[T, List[T]]): The value or list to ensure correct length.

        Returns:
            List[T]: A list with the correct number of elements.
        """
        if isinstance(value, (tuple, list)):
            assert len(value) == self.n_layers, f"Length of {type(value)} does not match number of layers: {len(value)} != {self.n_layers}"
            return value
        else:
            return [value] * self.n_layers


class FCBlockConfig:
    """
    Configuration for Fully Connected Block (FCBlock).

    This class provides a structured way to configure the options for building a fully connected block.

    Args:
        layers (List[int]): A sequence of integers specifying the number of units in each layer.
        dropout_rate (Union[float, List[float]], optional): Dropout rate(s) for each layer. Defaults to 0.0.
        use_batch_norm (Union[bool, List[bool]], optional): Whether to use batch normalization for each layer. Defaults to False.
        use_layer_norm (Union[bool, List[bool]], optional): Whether to use layer normalization for each layer. Defaults to False.
        activation_fn (Union[Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]], optional): Activation function(s) for each layer. Defaults to nn.ReLU.
        return_hidden (Union[bool, List[bool]], optional): Whether to aggregate and return hidden representations. Defaults to False.

    Attributes:
        n_layers (int): Number of layers in the configuration.
        layers (List[int]): List of layer sizes.
        dropout_rate (List[float]): List of dropout rates.
        use_batch_norm (List[bool]): List indicating whether to use batch normalization for each layer.
        use_layer_norm (List[bool]): List indicating whether to use layer normalization for each layer.
        activation_fn (List[Optional[Type[nn.Module]]]): List of activation functions.
        return_hidden (List[bool]): List indicating whether to return hidden layers.
    """

    def __init__(
        self,
        layers: List[int],
        dropout_rate: Union[float, List[float]] = 0.0,
        use_batch_norm: Union[bool, List[bool]] = False,
        use_layer_norm: Union[bool, List[bool]] = False,
        activation_fn: Union[Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]] = nn.ReLU,
        return_hidden: Union[bool, List[bool]] = False
    ):
        super().__init__()

        self._validate_and_set_layers(layers)
        self._validate_and_set_option('dropout_rate', dropout_rate, float)
        self._validate_and_set_option('use_batch_norm', use_batch_norm, bool)
        self._validate_and_set_option('use_layer_norm', use_layer_norm, bool)
        self._validate_and_set_option('activation_fn', activation_fn, nn.Module, comparison_fn=issubclass, optional=True)
        self._validate_and_set_option('return_hidden', return_hidden, bool)

    def _validate_and_set_layers(self, layers: List[int]):
        """
        Validate and set the layers configuration.

        Args:
            layers (List[int]): List of layer sizes.

        Raises:
            ValueError: If the layers are not a valid list of positive integers.
        """
        # Assert layers is a list
        assert isinstance(layers, list), f"layers must be a list found type: {type(layers)}"
        # Assert all values are integer objects greater than 0
        assert all(isinstance(layer, int) and layer > 0 for layer in layers), "layers must be positive integers"
        n_layers = len(layers)
        # Since layers will be paired into n_in and n_out
        # the number of layers will be equal to 1 for both
        # length of 1 and 2, otherwise the number of layers
        # will then be one less len(layers) because pairing
        # starting from the first two elements
        if n_layers > 1:
            n_layers -= 1
        self.n_layers = n_layers
        self.layers = layers
                
    def _validate_and_set_option(self, name: str, obj: Union[T, List[T]], req_type: Type, comparison_fn=isinstance, optional=False):
        """
        Validate and set an option configuration.

        Args:
            name (str): The name of the option.
            obj (Union[T, List[T]]): The option value or list of values.
            req_type (Type): Required type for validation.
            comparison_fn (Callable): Function to compare types (default is isinstance).
            optional (bool): Whether the option is optional.

        Raises:
            ValueError: If the option is not valid or the wrong type.
        """
        types = (req_type, type(None)) if optional else (req_type,)

        if not optional and obj is None:
            raise ValueError(f"{name} is not optional but value is None")

        if is_iterable(obj):
            if len(obj) != self.n_layers:
                raise ValueError(f"Length of '{name}' must match the length of 'layers': {len(obj)} != {self.n_layers}")
            try:
                assert all(val is not None and comparison_fn(val, types) for val in obj if not (optional and val is None))
            except (AssertionError, TypeError):
                raise ValueError(f"All elements in '{name}' must be a {str(req_type)}")
            value = obj
        elif obj is None or comparison_fn(obj, req_type):
            if optional and obj is None:
                value = None
            else:
                value = [obj] * self.n_layers
        else:
            raise ValueError(f"{name} ({obj}) is not a {str(req_type)} or iterable of {req_type}")

        setattr(self, name, value)
        
class FCBlock(BaseFCBlock):
    """Wrapper to intialize BaseFCBlock from FCBlockConfig"""
    
    def __init__(self, config: FCBlockConfig):
        super(FCBlock, self).__init__(
            layers=config.layers,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            activation_fn=config.activation_fn,
            return_hidden=config.return_hidden,
        )


class ConditionalLayer(nn.Module):
    """
    Conditionaly passes split input tensor through repespective condition layer.
    
    Args:
        batch_key (str): Column key in metadata.
        conditions_path (str): Path to unqiue conditions in dataset for batch_key (used to initialize modules).
        fc_block_config (FCBlockConfig): Configuration to be used for each condition module.
    """
    
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
    
    """
    Iterates over all ConditionLayer's allowing for shuffling or sequential iteration.
    
    Args:
        conditional_paths (dict[str, str]): Dictionary of batch_keys and paths to csv's with their respective unique conditions.
        fc_block_config (FCBlockConfig): Configuration for all ConditionLayer's.
        selection_order (Optional[list[str]]: Optional list of batch_keys to order forward pass on respective ConditionLayer's.
    """
    
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
        latent_dim (int): Size of latent dimension.
        fc_block_config (FCBlockConfig): Configuration for fc block.
        distribution (Union[Literal['ln'], Literal['normal']], optional): Type of distribution for the latent variables. Defaults to 'normal'.
        return_dist (bool, optional): Whether to return the distribution object. Defaults to False.
        var_eps (float, optional): Small epsilon value for numerical stability in variance calculation. Defaults to 1e-4.

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
        self.fc = FCBlock(fc_block_config)

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
    
    @property
    def n_layers(self):
        return self.fc.n_layers
    
    def encode(self, x: torch.Tensor):
        """Encode input and return output and list of hidden representations"""
        q, hidden_representations = self.fc(x)
        return q, hidden_representations
    
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
        q, hidden_representations = self.encode(x)
        
        # Compute the mean of the latent variables
        q_m = self.mean_encoder(q)
        # Compute the variance of the latent variables and add epsilon for numerical stability
        q_v = torch.exp(self.var_encoder(q)) + self.var_eps
        
        # Create a normal distribution with the computed mean and variance
        dist = Normal(q_m, q_v.sqrt())
        
        # Sample the latent variables and apply the transformation
        latent = self.z_transformation(dist.rsample())
        
        if self.return_dist:
            return dist, latent, hidden_representations
        
        return q_m, q_v, latent, hidden_representations


class BaseExpert(nn.Module):
    """
    Container that stores expert encoder and decoder networks.
    
    Args:
        id (str): Name of expert (unique identifier)
        encoder (FCBlock): encoder network
        decoder (FCBlock): decoder network 
        
    Attributes:
        id (str)
        encoder (FCBlock): encoder network
        decoder (FCBlock): decoder network 
    """
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
    """
    Wrapper to BaseExpert that allows easy configuration through FCBlockConfig.
    
    Args:
        id (str): Name of expert (unique identifier)
        encoder_config (FCBlockConfig): Encoder layer configuration
        decoder_config (FCBlockConfig): Decoder layer configuration
    
    Attributes:
        id (str)
        encoder (FCBlock): encoder network
        decoder (FCBlock): decoder network 
    """
    
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
    """
    Container to store Expert encoder and decoder networks.
    
    Args:
        experts (list[BaseExpert]): List of Expert modules.
        
    Attributes:
        labels (dict[str, int]): Dictionary of expert.id's to their integer representation.
    """
    
    def __init__(self, experts: list[BaseExpert]):
        super().__init__({ expert.id: expert for expert in experts})
        self.labels = { key: i for i, key in enumerate(self.keys())}