import torch
import torch.nn as nn
from sciml.utils.constants import REGISTRY_KEYS as RK

from .mixins.expert import ExpertMixIn
from .mixins.init import HeWeightInitMixIn

class Expert(ExpertMixIn, HeWeightInitMixIn, nn.Module):
    
    def __init__(
        self,
        encoder_layers,
        decoder_layers
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.build()
        self.init_weights()
        
    def build(self):
        """Initializes attributes encoder and decoder"""
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
    def build_encoder(self):
        layers = []
        n_in = self.encoder_layers[0]
        for n_out in self.encoder_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def build_decoder(self):
        layers = []
        n_in = self.decoder_layers[0]
        for n_out in self.decoder_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)