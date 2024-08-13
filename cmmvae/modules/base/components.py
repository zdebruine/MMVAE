from typing import Any, List, Literal, Optional, Type, TypeVar, Union
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

_BLOCK_CONFIG = {
    'dropout_rate': (float,),
    'use_batch_norm': (bool,),
    'use_layer_norm': (bool,),
    'activation_fn': (nn.Module, {'comparison_fn': issubclass, 'optional': True}),
    'return_hidden': (bool,)
}

class FCBlockConfig:
    """
    Configuration for Fully Connected Block :class:`FCBlock`.

    This class provides a structured way to configure the options for building a fully connected block.
    
    Attributes:
        n_layers (int): Number of layers in the configuration.
        layers (List[int]): List of layer sizes.
        dropout_rate (List[float]): List of dropout rates.
        use_batch_norm (List[bool]): List indicating whether to use batch normalization for each layer.
        use_layer_norm (List[bool]): List indicating whether to use layer normalization for each layer.
        activation_fn (List[Optional[Type[nn.Module]]]): List of activation functions.
        return_hidden (List[bool]): List indicating whether to return hidden layers.
    """
    
    layers: List[int]
    dropout_rate:  List[float]
    use_batch_norm: List[bool]
    use_layer_norm: List[bool]
    activation_fn: List[Optional[Type[nn.Module]]]
    return_hidden: List[bool]
    
    def __init__(
        self,
        layers: List[int],
        dropout_rate: Union[float, List[float]] = 0.0,
        use_batch_norm: Union[bool, List[bool]] = False,
        use_layer_norm: Union[bool, List[bool]] = False,
        activation_fn: Union[Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]] = None,
        return_hidden: Union[bool, List[bool]] = False
    ):
        r"""
        
        .. note::
            The FCBlock zips the layers into n_in, n_out pairings ([10, 20, 30] -> [(10, 20), (20, 30)]). If 
            a single layer size is provided, it's size is duplicated to form one pairing ([10] -> [(10, 10)]).
            
            Each attribute can be provided as a list for each layer pair 
            or a single value that is duplicated for all layers.

        Args:
            layers (List[int]): A sequence of integers specifying the number of units in each layer.
            dropout_rate (Union[float, List[float]], optional): Dropout rate(s) for each layer. Defaults to 0.0.
            use_batch_norm (Union[bool, List[bool]], optional): Whether to use batch normalization for each layer. Defaults to False.
            use_layer_norm (Union[bool, List[bool]], optional): Whether to use layer normalization for each layer. Defaults to False.
            activation_fn (Union[Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]], optional): Activation function(s) for each layer. Defaults to None.
            return_hidden (Union[bool, List[bool]], optional): Whether to aggregate and return hidden representations. Defaults to False.
        """
        super().__init__()
        
        # Assert layers is a list
        assert isinstance(layers, list), f"layers must be a list found type: {type(layers)}"
        # Assert all values are integer objects greater than 0
        assert all(isinstance(layer, int) and layer > 0 for layer in layers), "layers must be positive integers"
        if len(layers) == 1:
            layers = layers * 2
        self.layers = layers
        
        local_kwargs = locals()
        for name in _BLOCK_CONFIG:
            obj = local_kwargs[name]
            if not is_iterable(local_kwargs[name]):
                obj = [obj] * self.n_layers
            setattr(self, name, obj)
        
        self.validate()
    
    @property
    def n_layers(self):
        if not hasattr(self, 'layers'):
            raise RuntimeError("n_layers called before layers initialized")
        n_layers = len(self.layers)
        # Since layers will be paired into n_in and n_out
        # the number of layers will be equal to 1 for both
        # length of 1 and 2, otherwise the number of layers
        # will then be one less len(layers) because pairing
        # starting from the first two elements
        if n_layers > 1:
            n_layers -= 1
        return n_layers
    
    def _validate_option(self, name, req_type, comparison_fn=isinstance, optional=False):
        obj = getattr(self, name)
        if not optional and obj is None:
            raise ValueError(f"{name} is not optional but value is None")
        types = (req_type, type(None)) if optional else (req_type,)
        
        if len(obj) != self.n_layers:
            raise ValueError(f"Length of '{name}' must match the length of 'layers': {len(obj)} != {self.n_layers}")
        try:
            assert all(val is not None and comparison_fn(val, types) for val in obj if not (optional and val is None))
        except (AssertionError, TypeError):
            raise ValueError(f"All elements in '{name}' must be a {str(req_type)}")
        
    def validate(self):
        """Run validation over current configuration"""
        for name, (req_type, *kwargs) in _BLOCK_CONFIG.items():
            kwargs = kwargs[0] if kwargs else {}
            self._validate_option(name, req_type, **kwargs)


class FCBlock(nn.Module):
    """
    Fully Connected Block for building neural network layers.

    This class constructs a series of fully connected layers with optional dropout, batch normalization,
    layer normalization, and activation functions.

    Attributes:
        config (`FCBlockConfig`): Configuration class for FCBlock
        fc_layers (nn.Sequential): The sequential container of fully connected layers.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        can_bypass (bool): Whether we can rely on Sequential forward if not returning hidden states.
    """
    def __init__(self, config: FCBlockConfig):
        """Initilize Fully Connected Block from configuration."""
        super().__init__()
        
        # Validate config is compatible with FCBlockConfig
        self.config = config
        self.config.validate()
            
        # Create fully connected layers
        self.fc_layers = nn.Sequential(*[
            self._make_layer(
                n_in=n_in,
                n_out=n_out, 
                use_batch_norm=config.use_batch_norm[i], 
                use_layer_norm=config.use_layer_norm[i], 
                activation_fn=config.activation_fn[i], 
                dropout_rate=config.dropout_rate[i])
            for i, (n_in, n_out) in enumerate(zip(self.config.layers[:-1], self.config.layers[1:]))
        ])
    
    @property
    def input_dim(self) -> int:
        """The input dimension of the :class:`FCBlock`"""
        return self.config.layers[0]
    
    @property
    def output_dim(self) -> int:
        """The output dimension of the :class:`FCBlock`"""
        return self.config.layers[-1]
    
    @property
    def can_bypass(self) -> bool:
        """Whether module can bypass returning hidden states"""
        return not any(self.config.return_hidden)

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
                if name == '2' and self.config.return_hidden[i]:
                    hidden_representations.append(x)
        return x, hidden_representations


class ConditionalLayer(nn.Module):
    """
    Conditionaly passes split input tensor through repespective condition layer.
    
    .. note::
        To reduce loading time of creating and updating optimizer paramters on the fly the unique
        conditions prevelent in the dataset must be known in advance. :attr:`conditions_path` should 
        contain a path to a csv containing all the unique conditions. The batch_key is the column key
        in the metadata.
    
    Args:
        batch_key (str): Column key in metadata.
        conditions_path (str): Path to unqiue conditions in dataset for batch_key (used to initialize modules).
        fc_block_config (`FCBlockConfig`): Configuration to be used for each condition module.
    """
    def __init__(self, batch_key: str, conditions_path: str, fc_block_config: FCBlockConfig):
        """Initialize condition modules from condition paths"""
        super(ConditionalLayer, self).__init__()
        
        self.batch_key = batch_key
        self.unique_conditions = { self.format_condition_key(condition) for condition in pd.read_csv(conditions_path, header=None)[0] }
        
        self.conditions = nn.ModuleDict({
            condition: FCBlock(fc_block_config) 
            for condition in self.unique_conditions
        })
        
    def format_condition_key(self, condition: str) -> str:
        """
        Format condition key because '.' not allowed as `nn.ModuleDict` key.
        
        Args:
            condition (str): Condition key
        
        Returns:
            str: Condition key with all '.' replaced with '_'
        """
        return condition.replace('.', '_')
    
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, condition: Optional[str] = None):
        """
        Run forward pass through each metadata label specific condition layer. The metadata is 
        grouped by the batch_key and the indices are used to mask the input tensor. The resulting
        Tensors are summed to form the output batch maintaining original sample order.
        
        Args:
            x (torch.Tensor): Input tensor.
            metadata (pd.DataFrame): Metadata dataframe with corresponding batch_key
            condition (str, optional): If available, all samples passed through this condition.
        
        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
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
    
    Attributes:
        layers (nn.ModuleList): nn.ModuleList of :class:`ConditionLayer`'s
        selection_order (torch.Tensor): torch.Tensor representing selection order.
    """
    
    def __init__(
        self,
        conditional_paths: dict[str, str],
        fc_block_config: FCBlockConfig,
        selection_order: Optional[list[str]] = None,
    ):
        """
        Initialize :class:`ConditionLayer`'s for each batch_key.
        
        .. note::
            All :class:`ConditionLayer`'s are instantiated from the same configuration. 
            
            Each key in conditional_paths is a :class:`ConditionLayer`.
            
        Args:
            conditional_paths (dict[str, str]): Dictionary of batch_keys and paths to csv's with their respective unique conditions.
            fc_block_config (cmmvae.modules.base.FCBlockConfig): Configuration for all ConditionLayer's.
            selection_order (Optional[list[str]]: Optional list of batch_keys to order forward pass on respective ConditionLayer's.
        """
        super(ConditionalLayers, self).__init__()
        
        if not selection_order:
            selection_order = list(conditional_paths.keys())
            self.shuffle_selection_order = True
        else:
            self.shuffle_selection_order = False
            
        self.selection_order = torch.arange(0, len(selection_order), dtype=torch.int32, requires_grad=False)
        
        self.layers = nn.ModuleList([
            ConditionalLayer(batch_key, conditional_paths[batch_key], fc_block_config)
            for batch_key in selection_order
        ])
        
    def forward(self, x: torch.Tensor, metadata: pd.DataFrame):
        """
        Run forward pass through each :class:`ConditionLayer` either shuffled or in-order.
        
        Args:
            x (torch.Tensor): Input tensor
            metadata (pd.DataFrame): Input metadata
        
        Returns:
            torch.Tensor: Resulting torch.Tensor
        """
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
    Encoder module for a Variational Autoencoder (cmmvae.modules.VAE) with flexible configurations.

    Attributes:
        encoder (cmmvae.modules.base.FCBlock): Fully connected block for encoding input features.
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
        """
        Initialize Encoder nn.Module from configuration and set defaults.
        
        Args:
            latent_dim (int): Size of latent dimension.
            fc_block_config (`FCBlockConfig`): Configuration for fc block.
            distribution (Union[Literal['ln'], Literal['normal']], optional): Type of distribution for the latent variables. Defaults to 'normal'.
            return_dist (bool, optional): Whether to return the distribution object. Defaults to False.
            var_eps (float, optional): Small epsilon value for numerical stability in variance calculation. Defaults to 1e-4.
        """
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
    def n_layers(self) -> int:
        """Number of layer's."""
        return self.fc.config.n_layers
    
    def encode(self, x: torch.Tensor):
        """
        Encode input and return output and list of hidden representations
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        return self.fc(x)
    
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
        encoded = self.encode(x)
        if isinstance(encoded, tuple):
            q, hidden_representations = encoded
        else:
            hidden_representations = []
            q = encoded
        
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


class Expert(nn.Module):
    """
    Container that stores expert encoder and decoder networks.
        
    Attributes:
        id (str)
        encoder (`FCBlock`): encoder network
        decoder (`FCBlock`): decoder network 
    """
    def __init__(
        self,
        id: str,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig,
    ):
        """
        Initialize encoder and decoder network storage.
            
        Args:
            id (str): Name of expert (unique identifier)
            encoder (`FCBlock`): encoder network
            decoder (`FCBlock`): decoder network 
        """
        super().__init__()
        
        self.id = id
        self.encoder = FCBlock(encoder_config)
        self.decoder = FCBlock(decoder_config)
        
    def forward(self, *args, **kwargs):
        """
        .. warning::
            Forward pass will through NotImplementedErro as it does not make sense to pass through joint encoder and decoder.
        """
        raise NotImplementedError("Cannot call forward pass on disjoint encoder and decoder")
    
    def encode(self, x: torch.Tensor):
        """Run forward pass on encoder"""
        return self.encoder(x)
        
    def decode(self, x: torch.Tensor):
        """Run forward pass on decoder"""
        return self.decoder(x)

class Experts(nn.ModuleDict):
    """
    Container to store cmmvae.modules.base.Expert encoder and decoder networks.
    
    Args:
        experts (list[cmmvae.modules.base.BaseExpert]): List of Expert modules.
        
    Attributes:
        labels (dict[str, int]): Dictionary of expert.id's to their integer representation.
    """
    
    def __init__(self, experts: list[Expert]):
        super().__init__({ expert.id: expert for expert in experts})
        self.labels = { key: i for i, key in enumerate(self.keys())}