import torch.nn as nn

def _submodules_init_weights_xavier_uniform_(module: nn.Module):
    for name, module in module.named_modules():
        _init_weights_xavier_uniform_(module, name)

def _init_weights_xavier_uniform_(module: nn.Module, name = None):
    if isinstance(module, nn.Linear):
        name = name if name is not None else module.__name__
        print(f"Initializing {name} with xaviar_uniform_")

def _xavier_uniform_(module: nn.Linear, bias = 0.0):
    nn.init.xavier_uniform_(module.weight)
    nn.init.constant_(module.bias, bias)