import torch
import torch.nn as nn
from typing import Iterable, Union, Optional, Any
from ._fc_block import FCBlock, FCBlockConfig
from sciml.constants import REGISTRY_KEYS as RK



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
    """
    A module dictionary for managing human and mouse experts.

    Args:
        human_encoder_kwargs (dict[str, Any]): Keyword arguments for the human encoder.
        human_decoder_kwargs (dict[str, Any]): Keyword arguments for the human decoder.
        mouse_encoder_kwargs (dict[str, Any]): Keyword arguments for the mouse encoder.
        mouse_decoder_kwargs (dict[str, Any]): Keyword arguments for the mouse decoder.
    """
    
    def __init__(self, *experts: BaseExpert):
        super().__init__()
        
        self.experts = nn.ModuleDict({
            expert.id: expert
            for expert in experts
        })