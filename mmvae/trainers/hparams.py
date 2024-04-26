import copy

class _HPConfigMeta(type):
    def __new__(cls, name, bases, attrs):
        # Aggregate required_hparams from all bases
        aggregated_hparams = {}
        for base in reversed(bases):  # Reverse to maintain MRO
            if hasattr(base, 'required_hparams'):
                aggregated_hparams.update(base.required_hparams)
        # Update with the current class's __required_hparams, if any
        aggregated_hparams.update(attrs.get('required_hparams', {}))
        attrs['_aggregated_hparams'] = aggregated_hparams
        return super().__new__(cls, name, bases, attrs)

class HPConfig(dict, metaclass=_HPConfigMeta):
    """
    The `HPConfig` class serves as a dictionary for hyperparameters (hparams) for pytorch. 
    It provides functionalities to load, access, validate, and manipulate configurations stored in a JSON file. 
    The dictionary is then flattened to be incomplience with pytorch by dot notation.
    
    Init:
        config_path (str): Path to json file to load into config.
        
        modifier (lambda dict: dict): lambda to modify the config loaded before flattening.
        
    Attributes:
        required_hparams (dict): A class attribute that stores the required hyperparameters along with their expected types. 
                                    This dictionary aggregates all subclassed required_hparams.

        config (dict): Holds the configuration parameters loaded from a JSON file.
        
    Structure:
        { "parent: { "child": value } } => { "parent.child": value }
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.required_hparams = self._aggregated_hparams
        self.__parse_hparams()
        self.__validate_hparams()
        self._initialized = True
        
    def __setattr__(self, __name: str, __value) -> None:
        if self.__dict__.get("_initialized", False):
            if __name in ('config',):
                raise RuntimeError(f"Attribute {__name} cannot be set after runtime.")
        super().__setattr__(__name, __value)
    
    def __parse_hparams(self):
        config = copy.deepcopy(self.config)
        return self.__flatten_dict(config)
        
    def __flatten_dict(self, _dict: dict, parent = ""):
        """Flattens the dictionary provided and adds all attributes to self"""
        for key, value in _dict.items():
            flattened_key = f"{parent}.{key}" if parent else key
            if isinstance(value, dict):
                self.__flatten_dict(value, flattened_key)
            else:
                if value == None:
                    value = self.required_hparams[flattened_key]() if flattened_key in self.required_hparams else None
                self[flattened_key] = value
    
    def __validate_hparams(self):
        for _req_key, _req_type in self.required_hparams.items():
            if _req_key not in self:
                raise ValueError(f"Required hparam for HPBaseTrainer {_req_key} not in supplied hparams!")
            _hp = self[_req_key]
            if _hp is None or not isinstance(_hp, _req_type):
                raise TypeError(f"Required hparam for HPBaseTrainer {_req_key} expected type {_req_type} but received {type(_hp)}")
