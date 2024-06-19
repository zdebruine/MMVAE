import torch
import torch.nn as nn
from sciml.utils.constants import REGISTRY_KEYS as RK

from .mixins.mmvae import MMVAEMixIn


class MMVAE(MMVAEMixIn, nn.Module):
    
    def __init__(
        self,
        vae,
        human_expert,
        mouse_expert
    ):
        super().__init__()
        self.vae = vae
        self.experts = nn.ModuleDict({RK.HUMAN: human_expert, RK.MOUSE: mouse_expert})
    
    def configure_optimizers(self):
        return (torch.optim.Adam(self.vae.parameters()),
                torch.optim.Adam(self.experts[RK.HUMAN].parameters()),
                torch.optim.Adam(self.experts[RK.MOUSE].parameters()))