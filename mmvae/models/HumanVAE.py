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
            print("Initialing SharedVAE xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True
        

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after intialization!")
        
        utils._submodules_init_weights_xavier_uniform_(self)
        
class HumanExpert(M.Expert):

    _initialized = None

    def __init__(self, encoder, decoder, init_weights = False):
        super(HumanExpert, self).__init__(encoder, decoder)

        if init_weights:
            print("Initialing HumanExpert xavier uniform on all submodules")
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
    
class HumanEncoder(nn.Module):
    
    def __init__(self, writer, drop_out=False):
        super().__init__()
        self.fc1 = nn.Linear(60664, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.writer = writer
        self.droput = drop_out
        
    def apply_dropout(self, x):
        if self.__getattribute__('_iter') is None: 
            self._iter = 0
        else:
            self._iter += 1
            
        fc1_dp = max(0.8 - (self._iter * (1 / 5e4)), 0.3)
        x = F.dropout(x, p=fc1_dp)
        self.writer.add_scalar('Metric/fc1_dp', fc1_dp, self._iter)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        
        if self.droput:
            x = self.apply_dropout(x)  
            
        x = F.relu(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class SharedEncoder(nn.Module):

    _initialized = None

    def __init__(self):
        super(SharedEncoder, self).__init__()
        #self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, x):
        #x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class SharedDecoder(nn.Module):

    def __init__(self):
        super(SharedDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        #self.fc3 = nn.Linear(512, 512)
        #self.fc4 = nn.Linear(512, 512)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        #x = F.leaky_relu(self.fc4(x))
        return x
    
def configure_model(device, writer) -> Model:
    return Model(
            HumanExpert(
                nn.Sequential(
                    nn.Linear(60664, 512),
                    nn.ELU(),
                    nn.Linear(512, 256),
                    nn.ELU()
                    ),
                nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ELU(),
                    nn.Linear(512, 60664),
                    nn.ELU()
                ),
                init_weights=False
            ),
            SharedVAE(
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ELU(),
                    nn.Linear(128, 64),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ELU(),
                    nn.Linear(64, 128),
                    nn.ELU(),
                    nn.Linear(128, 256),
                    nn.ELU()
                    ),
                nn.Linear(64, 32),
                nn.Linear(64, 32),
                init_weights=False
            )
        ).to(device)

        
