import torch
import torch.nn as nn
import torch.nn.functional as F
import mmvae.models as M
import mmvae.models.utils as utils

class SharedVAE(M.VAE):

    _initialized = None

    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear, init_weights = False):
        super(SharedVAE, self).__init__(encoder, decoder, mean, var)

        if init_weights:
            print("Initialing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True
        

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after intialization!")
        
        utils._submodules_init_weights_xavier_uniform_(self)

class HumanEncoder(nn.Module):
    
    def __init__(self, writer):
        super().__init__()
        self.fc1 = nn.Linear(60664, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 768)
        self._iter = 0
        self.writer = writer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._iter += 1
            
        #fc1_dp = max(0.8 - (self._iter * (1 / 2e4)), 0.3)
        
        #self.writer.add_scalar('Metric/fc1_dp', fc1_dp, self._iter)
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
        
class HumanExpert(M.Expert):

    _initialized = None

    def __init__(self, encoder, decoder, init_weights = False):
        super(HumanExpert, self).__init__(encoder, decoder)

        if init_weights:
            print("Initialing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after intialization!")
        utils._submodules_init_weights_xavier_uniform_(self)

class Model(nn.Module):

    def __init__(self, expert: M.Expert, shared_vae: M.VAE):
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
    
def configure_model(device, writer) -> Model:
        return Model(
            HumanExpert(
                HumanEncoder(writer),
                nn.Sequential(
                    nn.Linear(768, 768),
                    nn.LeakyReLU(),
                    nn.Linear(768, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 60664),
                    nn.ReLU()
                ),
                init_weights=True
            ),
            SharedVAE(
                SharedEncoder(),
                SharedDecoder(),
                nn.Linear(256, 128),
                nn.Linear(256, 128),
                init_weights=True
            )
        ).to(device)
