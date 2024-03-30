import torch
import torch.nn as nn
import torch.nn.functional as F
import mmvae.models as M
import mmvae.models.utils as utils
from typing import Tuple, Union, Any

class SharedVAE_FT(nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear, finetune: nn.Linear) -> None:
        super(SharedVAE_FT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        self.finetune = finetune
        
        self.finetune.weight.data.fill_(0.0)
        self.finetune.bias.data.fill_(0.0)
        
    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps

    def forward(self, x: torch.Tensor) -> Tuple[Union[Any, torch.Tensor], Any, Any, Tuple[Any]]:
        x, *encoder_results = utils.parameterize_returns(self.encoder(x))
        
        mu = self.mean(x)
        var = self.var(x)
                
        x = self.reparameterize(mu, torch.exp(0.5 * var))
        
        # Fine-tuning layer
        x = self.finetune(x)
        
        x, *decoder_results = utils.parameterize_returns(self.decoder(x))
            
        return (x, mu, var, *encoder_results, *decoder_results)
        
class HumanExpert(M.Expert):
    
    def __init__(self, encoder, decoder):
        super(HumanExpert, self).__init__(encoder, decoder)

class Model(nn.Module):

    def __init__(self, expert: M.Expert, shared_vae: SharedVAE_FT):
        super().__init__()
        
        self.expert = expert
        self.shared_vae = shared_vae

    def forward(self, x: torch.Tensor):
        x = self.expert.encoder(x)
        x, mu, var = self.shared_vae(x)
        x = self.expert.decoder(x)
        return x, mu, var

class SharedEncoder(nn.Module):

    _initialized = None

    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class SharedDecoder(nn.Module):

    def __init__(self):
        super(SharedDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 768)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return x
    
def configure_model() -> Model:
        return Model(
            HumanExpert(
                nn.Sequential(
                    nn.Linear(60664, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 768),
                    nn.LeakyReLU(),
                    nn.Linear(768, 768),
                    nn.LeakyReLU(),
                ),
                nn.Sequential(
                    nn.Linear(768, 768),
                    nn.LeakyReLU(),
                    nn.Linear(768, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 60664),
                    nn.LeakyReLU()
                )
            ),
            SharedVAE_FT(
                SharedEncoder(),
                SharedDecoder(),
                nn.Linear(256, 128),
                nn.Linear(256, 128),
                nn.Linear(128, 128),
            )
        )