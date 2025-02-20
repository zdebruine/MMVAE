import os
from typing import Callable, List, Literal, Optional, Type, TypeVar, Union
from collections import OrderedDict, defaultdict
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal


T = TypeVar("T")


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
    "dropout_rate": (float,),
    "use_batch_norm": (bool,),
    "use_layer_norm": (bool,),
    "activation_fn": (nn.Module, {"comparison_fn": issubclass, "optional": True}),
    "return_hidden": (bool,),
}


class FCBlockConfig:
    """
    Configuration for Fully Connected Block :class:`FCBlock`.

    This class provides a structured way to configure the options
        for building a fully connected block.

    Attributes:
        n_layers (int): Number of layers in the configuration.
        layers (List[int]): List of layer sizes.
        dropout_rate (List[float]): List of dropout rates.
        use_batch_norm (List[bool]):
            List indicating whether to use batch normalization for each layer.
        use_layer_norm (List[bool]):
            List indicating whether to use layer normalization for each layer.
        activation_fn (List[Optional[Type[nn.Module]]]):
            List of activation functions.
        return_hidden (List[bool]):
            List indicating whether to return hidden layers.
    """

    layers: List[int]
    dropout_rate: List[float]
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
        return_hidden: Union[bool, List[bool]] = False,
        activation_fn: Union[
            Optional[Type[nn.Module]], List[Optional[Type[nn.Module]]]
        ] = None,
    ):
        r"""

        .. note::
            The FCBlock zips the layers into n_in, n_out pairings
            ([10, 20, 30] -> [(10, 20), (20, 30)]). If a single layer
            size is provided, it's size is duplicated to form one
            pairing ([10] -> [(10, 10)]).

            Each attribute can be provided as a list for each layer pair
            or a single value that is duplicated for all layers.

        Args:
            layers (List[int]):
                A sequence of integers specifying the number of
                    units in each layer.
            dropout_rate (Union[float, List[float]], optional):
                Dropout rate(s) for each layer. Defaults to 0.0.
            use_batch_norm (Union[bool, List[bool]], optional):
                Whether to use batch normalization for each layer.
                    Defaults to False.
            use_layer_norm (Union[bool, List[bool]], optional):
                Whether to use layer normalization for each layer.
                    Defaults to False.
            activation_fn (Union[Optional[Type[nn.Module]],
                        List[Optional[Type[nn.Module]]]], optional):
                Activation function(s) for each layer. Defaults to None.
            return_hidden (Union[bool, List[bool]], optional):
                Whether to aggregate and return hidden representations.
                    Defaults to False.
        """
        super().__init__()

        # Assert layers is a list
        try:
            assert isinstance(layers, list)
        except AssertionError:
            raise ValueError(f"layers must be a list found type: {type(layers)}")
        # Assert all values are integer objects greater than 0
        try:
            assert all(isinstance(layer, int) and layer > 0 for layer in layers)
        except AssertionError:
            raise ValueError("layers must be positive integers")
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
        if not hasattr(self, "layers"):
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

    def _validate_option(
        self, name, req_type, comparison_fn=isinstance, optional=False
    ):
        obj = getattr(self, name)
        if not optional and obj is None:
            raise ValueError(f"{name} is not optional but value is None")
        types = (req_type, type(None)) if optional else (req_type,)

        if len(obj) != self.n_layers:
            raise ValueError(
                f"Length of '{name}' must match the length of 'layers':"
                f"{len(obj)} != {self.n_layers}"
            )
        try:
            assert all(
                val is not None and comparison_fn(val, types)
                for val in obj
                if not (optional and val is None)
            )
        except (AssertionError, TypeError):
            raise ValueError(f"All elements in '{name}' must be a {str(req_type)}")

    def validate(self):
        """Run validation over current configuration"""
        for name, (req_type, *kwargs) in _BLOCK_CONFIG.items():
            kwargs = kwargs[0] if kwargs else {}
            self._validate_option(name, req_type, **kwargs)


class ConcatBlockConfig(FCBlockConfig):
    def __init__(
        self,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        return_hidden: bool = False,
        activation_fn: Optional[Type[nn.Module]] = None,
    ):
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.return_hidden = return_hidden
        self.activation_fn = activation_fn


class FCBlock(nn.Module):
    """
    Fully Connected Block for building neural network layers.

    This class constructs a series of fully connected layers with optional
    dropout/batch/layer normalization, and activation functions.

    Attributes:
        config (`FCBlockConfig`): Configuration class for FCBlock
        fc_layers (nn.Sequential):
            The sequential container of fully connected layers.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        can_bypass (bool): Whether we can rely on Sequential forward
            if not returning hidden states.
    """

    def __init__(self, config: FCBlockConfig):
        """Initilize Fully Connected Block from configuration."""
        super().__init__()

        # Validate config is compatible with FCBlockConfig
        config.validate()
        self.config = config

        layers = [
            self._make_layer(
                n_in=n_in,
                n_out=n_out,
                use_batch_norm=config.use_batch_norm[i],
                use_layer_norm=config.use_layer_norm[i],
                activation_fn=config.activation_fn[i],
                dropout_rate=config.dropout_rate[i],
                return_hidden=config.return_hidden[i],
            )
            for i, (n_in, n_out) in enumerate(
                zip(self.config.layers[:-1], self.config.layers[1:])
            )
        ]

        self.fc_layers = nn.Sequential(*layers)

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

    def _make_layer(
        self,
        n_in: int,
        n_out: int,
        use_batch_norm: bool,
        use_layer_norm: bool,
        activation_fn: Optional[Type[nn.Module]],
        dropout_rate: float,
        return_hidden: bool,
    ) -> nn.Sequential:
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
            nn.Sequential: A sequence of layers comprising
                the fully connected layer.
        """
        layers = OrderedDict()
        layers["lin"] = nn.Linear(n_in, n_out)

        if use_batch_norm:
            layers["bn"] = nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
        if use_layer_norm:
            layers["ln"] = nn.LayerNorm(n_out, elementwise_affine=False)
        if activation_fn is not None:
            if issubclass(activation_fn, nn.Softmax):
                layers["af"] = activation_fn(dim=1)
            else:
                layers["af"] = activation_fn()
        if dropout_rate > 0:
            layers["dr"] = nn.Dropout(p=dropout_rate)

        return nn.Sequential(layers)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """
        Forward pass through the fully connected block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
                Output tensor or tuple of output andhidden representations.
        """
        if self.can_bypass:
            return self.fc_layers(x)

        hidden_representations = []
        for i, layer in enumerate(self.fc_layers):
            for name, sublayer in layer.named_children():
                x = sublayer(x)
                if name == "af" and self.config.return_hidden[i]:
                    hidden_representations.append(x)
        return x, hidden_representations


class ConditionalLayer(nn.Module):
    """
    Conditionaly passes split input tensor through
        repespective condition layer.

    .. note::
        To reduce loading time of creating and updating optimizer
        paramters on the fly the unique conditions prevelent in the
        dataset must be known in advance. :attr:`conditions_path` should
        contain a path to a csv containing all the unique conditions.
        The batch_key is the column key in the metadata.

    Args:
        batch_key (str): Column key in metadata.
        conditions_path (str):
            Path to unqiue conditions in dataset for batch_key
                (used to initialize modules).
        fc_block_config (`FCBlockConfig`):
            Configuration to be used for each condition module.
    """

    def __init__(
        self, batch_key: str, conditions_path: str, fc_block_config: FCBlockConfig
    ):
        """Initialize condition modules from condition paths"""
        super(ConditionalLayer, self).__init__()

        self.batch_key = batch_key

        self.conditions = nn.ModuleDict(
            {
                self.format_condition_key(condition): FCBlock(fc_block_config)
                for condition in pd.read_csv(conditions_path, header=None)[0]
            }
        )

    def format_condition_key(self, condition: str) -> str:
        """
        Format condition key because '.' not allowed as `nn.ModuleDict` key.

        Args:
            condition (str): Condition key

        Returns:
            str: Condition key with all '.' replaced with '_'
        """
        return condition.replace(".", "_")

    def forward(
        self, x: torch.Tensor, metadata: pd.DataFrame, condition: Optional[str] = None
    ):
        """
        Run forward pass through each metadata label-specific condition layer.
        The metadata is used to index the input tensor based on conditions.
        The resulting tensors are combined to form the output batch,
        maintaining the original sample order.

        Args:
            x (torch.Tensor):
                Input tensor.
            metadata (pd.DataFrame):
                Metadata dataframe with corresponding batch_key.
            condition (str, optional):
                If provided, all samples are passed through this condition.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        if condition:
            condition = self.format_condition_key(condition)
            return self.conditions[condition](x)

        device = x.device

        # Extract condition keys for the batch
        condition_keys = (
            metadata[self.batch_key]
            .astype(str)
            .apply(self.format_condition_key)
            .tolist()
        )

        # Map conditions to sample indices
        condition_to_indices = {}
        for idx, cond_key in enumerate(condition_keys):
            condition_to_indices.setdefault(cond_key, []).append(idx)

        xhat = torch.empty_like(x)

        # Process samples for each condition in a batch
        for cond_key, indices in condition_to_indices.items():
            indices_tensor = torch.tensor(indices, device=device)
            x_cond = x.index_select(0, indices_tensor)
            x_cond_processed = self.conditions[cond_key](x_cond)
            xhat.index_copy_(0, indices_tensor, x_cond_processed)

        return xhat


def _is_valid_file(fname, batch_key):
    return fname == f"unique_expression_{batch_key}.csv"


def collect_species_files(
    directory,
    batch_keys,
    species_files=None,  # Holds species names and their batch_keys with file paths
    is_valid_file: Optional[Callable[[str, str], bool]] = None,
):
    if is_valid_file is None:
        is_valid_file = _is_valid_file

    if species_files is None:
        species_files = {}

    # Collect batch_keys present in 'shared' directory
    shared_dir = os.path.join(directory, "shared")
    shared_files = {}
    if os.path.isdir(shared_dir):
        for fname in os.listdir(shared_dir):
            file_path = os.path.join(shared_dir, fname)
            if os.path.isfile(file_path):
                for batch_key in batch_keys:
                    if is_valid_file(fname, batch_key):
                        shared_files[batch_key] = file_path
                        break  # Stop checking other batch_keys for this file

    species_files["shared"] = shared_files

    # Collect batch_keys for each species excluding 'shared'
    for entry in os.listdir(directory):
        species_dir_path = os.path.join(directory, entry)
        if os.path.isdir(species_dir_path) and entry != "shared":
            species_name = entry
            species_specific_files = {}
            for fname in os.listdir(species_dir_path):
                file_path = os.path.join(species_dir_path, fname)
                if os.path.isfile(file_path):
                    for batch_key in batch_keys:
                        if is_valid_file(fname, batch_key):
                            # Only add if batch_key is not in shared
                            if batch_key not in shared_files:
                                species_specific_files[batch_key] = file_path
                            break  # Stop checking other batch_keys for this file
            if species_specific_files:
                species_files[species_name] = species_specific_files
    print(f"Collected species files {species_files}")
    return species_files


class ConditionalLayers(nn.Module):

    """
    Iterates over all ConditionLayer's allowing for shuffling
        or sequential iteration.

    Attributes:
        layers (nn.ModuleList): nn.ModuleList of :class:`ConditionLayer`'s
        selection_order (torch.Tensor):
            torch.Tensor representing selection order.
    """

    def __init__(
        self,
        directory: str,
        conditionals: list[str],
        fc_block_config: FCBlockConfig,
        selection_order: Optional[list[str]] = None,
    ):
        """
        Initialize :class:`ConditionLayer`'s for each batch_key.

        .. note::
            All :class:`ConditionLayer`'s are instantiated
                from the same configuration.

            Each key in conditional_paths is a :class:`ConditionLayer`.

        Args:
            conditional_paths (dict[str, str]):
                Dictionary of batch_keys and paths to csv's
                    with their respective unique conditions.
            fc_block_config (cmmvae.modules.base.FCBlockConfig):
                Configuration for all ConditionLayer's.
            selection_order (Optional[list[str]]:
                Optional list of batch_keys to order forward pass
                    on respective ConditionLayer's.
        """
        super(ConditionalLayers, self).__init__()

        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Could not intialize the conditional layers either due to the directory not existing yet\n{directory}"
            )
        # Prevent parsing the species conditional as no conditional layer is needed
        conditionals.remove("species")
        conditional_paths = collect_species_files(directory, conditionals)
        conditionals.append("species")

        self.shared_conditionals = list(conditional_paths["shared"].keys())

        self.shuffle_selection_order = False
        self.is_parallel = selection_order[0] == "parallel"
        if not selection_order or self.is_parallel:
            selection_order = conditionals
            self.shuffle_selection_order = True

        # Add all shared conditional layers
        layer_dict = {
            batch_key: ConditionalLayer(
                batch_key,
                conditional_paths["shared"][batch_key],
                fc_block_config,
            )
            for batch_key in conditional_paths["shared"]
        }

        # Find all species specific conditionals
        species_specific = {}
        for species in conditional_paths:
            if species == "shared":
                continue
            for batch_key in conditional_paths[species]:
                if not species_specific.get(batch_key):
                    species_specific[batch_key] = {}
                species_specific[batch_key].update(
                    {species: conditional_paths[species][batch_key]}
                )

        for batch_key in species_specific:
            if batch_key in layer_dict:
                raise RuntimeError(
                    f"batch_key '{batch_key}' is shared but attempted to make species specific"
                )

            layer_dict.update(
                {
                    batch_key: nn.ModuleDict(
                        {
                            species: ConditionalLayer(
                                batch_key,
                                conditions_path,
                                fc_block_config,
                            )
                            for species, conditions_path in species_specific[
                                batch_key
                            ].items()
                        }
                    )
                }
            )

        if "species" in conditionals:
            assert "species" not in layer_dict
            layer_dict.update(
                {
                    "species": nn.ModuleDict(
                        {
                            species: FCBlock(fc_block_config)
                            for species in conditional_paths
                            if species != "shared"
                        }
                    )
                }
            )

        self.layers = nn.ModuleDict(layer_dict)
        self.selection_order = selection_order

    def forward(
        self, x: torch.Tensor, metadata: pd.DataFrame, species: Optional[str] = None
    ):
        """
        Run forward pass through each ConditionLayer
        either shuffled or in-order.

        Args:
            x (torch.Tensor): Input tensor
            metadata (pd.DataFrame): Input metadata
            species (Optional[str]): Species identifier for species-specific layers.

        Returns:
            torch.Tensor: Resulting tensor after applying the layers.
        """

        # Determine the selection order
        if self.shuffle_selection_order:
            # Shuffle the selection order using Python's random module
            order = random.sample(self.selection_order, len(self.selection_order))
        else:
            order = self.selection_order

        xs = []
        # Apply each layer in the determined order
        for conditional in order:
            layer = self.layers[conditional]
            if isinstance(layer, nn.ModuleDict):
                if species is None:
                    raise RuntimeError(
                        f"'species' must be set to access non-shared conditional layer for batch_key '{conditional}'"
                    )
                layer = layer[species]
            if isinstance(layer, ConditionalLayer):
                if self.is_parallel:
                    xs.append(layer(x, metadata))
                else:
                    x = layer(x, metadata)
            else:
                if self.is_parallel:
                    xs.append(layer(x))
                else:
                    x = layer(x)
        if xs:
            x = torch.cat(xs, dim=1)
        return x


def _identity(x):
    return x


class Adversarial(nn.Module):
    """
    """

    labels = defaultdict(dict)

    def __init__(
        self,
        encoder: FCBlockConfig,
        heads: FCBlockConfig,
        conditions: list[str],
        labels_dir: str,
    ):
        super().__init__()
        self.encoder = FCBlock(encoder)
        head_nodes = {}
            
        for condition in conditions:
            df = pd.read_csv(os.path.join(labels_dir, f"human/unique_expression_{condition}.csv"), header=None)
            if condition not in Adversarial.labels.keys():
                for idx, value in enumerate(df[0]):
                    Adversarial.labels[condition][value] = idx
            
            heads.layers = [self.encoder.output_dim, len(df)]
            head_nodes[condition] = FCBlock(heads)

        self.heads = nn.ModuleDict(head_nodes)
    
    def forward(self, x: torch.Tensor):
        xhat = self.encoder(x)

        predictions = {}

        for category, layer in self.heads.items():
            predictions[category] = layer(xhat)

        return predictions

class Encoder(nn.Module):
    """
    Encoder module for a Variational Autoencoder (cmmvae.modules.VAE)
        with flexible configurations.

    Attributes:
        encoder (cmmvae.modules.base.FCBlock):
            Fully connected block for encoding input features.
        mean_encoder (nn.Linear):
            Linear layer to compute the mean of the latent variables.
        var_encoder (nn.Linear):
            Linear layer to compute the variance of the latent variables.
        z_transformation (Callable):
            Transformation applied to the latent variables.
                Defaults to softmax for log-normal distribution.
        var_activation (Callable):
            Activation function for the variance. Defaults to torch.exp.
        var_eps (float): Small epsilon value for numerical stability.
        return_dist (bool): Whether to return the distribution object.
    """

    def __init__(
        self,
        latent_dim: int,
        fc_block_config: FCBlockConfig,
        distribution: Union[Literal["ln"], Literal["normal"]] = "normal",
        return_dist: bool = False,
        hidden_z: bool = False,
        var_eps: float = 1e-4,  # numerical stability
    ):
        """
        Initialize Encoder nn.Module from configuration and set defaults.

        Args:
            latent_dim (int): Size of latent dimension.
            fc_block_config (`FCBlockConfig`): Configuration for fc block.
            distribution (Union[Literal['ln'], Literal['normal']], optional):
                Type of distribution for the latent variables.
                    Defaults to 'normal'.
            return_dist (bool, optional): Return the distribution object.
                Defaults to False.
            hidden_z (bool, optional): Return z as a hidden representation.
                Defaults to False.
            var_eps (float, optional):
                Small epsilon value for numerical stability
                    in variance calculation. Defaults to 1e-4.
        @reference: Heavily inspired by scvi-encoder at:
            https://docs.scvi-tools.org/en/stable/api/reference/scvi.nn.Encoder.html
        """
        super().__init__()
        self.fc = FCBlock(fc_block_config)

        # Get hidden and latent dimenision from layer dim list
        n_hidden = fc_block_config.layers[-1]

        # Linear layer to compute the mean of the latent variables
        self.mean_encoder = nn.Linear(n_hidden, latent_dim)

        # Linear layer to compute the variance of the latent variables
        self.var_encoder = nn.Linear(n_hidden, latent_dim)

        # Transformation for latent variables
        # (softmax for log-normal distribution, identity otherwise)
        self.z_transformation = _identity
        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        # Small epsilon value for numerical stability
        self.var_eps = var_eps

        # Whether to return the distribution object
        self.return_dist = return_dist

        # Whether to return the latent vector as a hidden representation
        self.hidden_z = hidden_z

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
            Union[Tuple[torch.distributions.Normal, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If return_dist is True,
                returns the distribution and latent variables.
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

        # Compute the variance of the latent variables
        # and add epsilon for numerical stability
        q_v = torch.exp(self.var_encoder(q)) + self.var_eps

        # Create a normal distribution with the computed mean and variance
        dist = Normal(q_m, q_v.sqrt())

        # Sample the latent variables and apply the transformation
        latent = self.z_transformation(dist.rsample())

        if self.hidden_z:
            hidden_representations.append(latent)

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
            Forward pass will through NotImplementedError
            as it does not make sense to pass through
            joint encoder and decoder.
        """
        raise NotImplementedError(self.forward.__doc__)

    def encode(self, x: torch.Tensor):
        """Run forward pass on encoder"""
        return self.encoder(x)

    def decode(self, x: torch.Tensor):
        """Run forward pass on decoder"""
        return self.decoder(x)


class Experts(nn.ModuleDict):
    """
    Container to store cmmvae.modules.base.Expert
        encoder and decoder networks.

    Args:
        experts (list[cmmvae.modules.base.BaseExpert]):
            List of Expert modules.

    Attributes:
        labels (dict[str, int]):
            Dictionary of expert.id's to their integer representation.
    """

    def __init__(self, experts: list[Expert]):
        super().__init__({expert.id: expert for expert in experts})
        self.labels = {key: i for i, key in enumerate(self.keys())}


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer as introduced in [Ganin2016]_.

    Implementation from GitHub: fungtion/DANN
    Specifically: https://github.com/fungtion/DANN/blob/476147f70bb818a63bb3461a6ecc12f97f7ab15e/models/functions.py

    Reference Source: https://github.com/ohlerlab/liam/blob/main/liam/_mymodule.py#L150
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
