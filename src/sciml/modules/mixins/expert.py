import torch
from sciml.utils.constants import REGISTRY_KEYS as RK

class ExpertMixIn:
    """
    Defines Expert forward pass.
    Expectes encoder and decoder to be defined
    """
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self):
        raise NotImplementedError("The Expert objects should not be called directly!!!")