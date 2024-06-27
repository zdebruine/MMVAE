from typing import Any
import torch
import torch.nn as nn


class BaseModule(nn.Module):
    
    def get_module_inputs(self, batch):
        """returns positional arguments to be passed to forward pass"""
        
    def loss(self, model_inputs, model_outputs, **kwargs) -> dict[str, Any]:
        """Compute losses"""
    
    def configure_optimizers(self):
        """Configure module optimizers for training"""
    
    def optimize(self, optimizers):
        """Used to optimize model if not using automatic optimization"""
        raise NotImplementedError("Optimize not implemented")
