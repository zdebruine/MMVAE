import torch.nn as nn
import torch

def _submodules_init_weights_xavier_uniform_(module: nn.Module, bias = None):
    for name, module in module.named_modules():
        if bias:
            _xavier_uniform_(module, bias)
        else:
            _xavier_uniform_(module)

def _xavier_uniform_(module: nn.Module, bias = 0.0):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, bias)

def parameterize_returns(results):
    if isinstance(results, tuple):
        if len(results) >= 1:
            x = results[0]
            results = results[1:]
        else:
            raise RuntimeError("VAE encoder returned a tuple with zero elements!")
    elif isinstance(results, torch.Tensor):
        x = results
        results = ()
    else:
        raise RuntimeError("The only supported argument results from encoder_results is torch.Tensor or tuple[torch.Tensor, *args]")
    return (x, *results)