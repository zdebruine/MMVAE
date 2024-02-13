import torch
import torch.nn as nn
import torch.nn.functional as F
import mmvae.models as M
import mmvae.models.utils as utils

class SharedVAE(M.VAE):

    def __init__(self, *args, **kwargs):
        super(SharedVAE, self).__init__(*args, **kwargs)
        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)
        utils._submodules_init_weights_xavier_uniform_(self.mean)
        utils._xavier_uniform_(self.var, -1.0) # TODO: Add declare why
    
class HumanExpert(M.Expert):

    def __init__(self, *args, **kwargs):
        super(HumanExpert, self).__init__(*args, **kwargs)
        utils._submodules_init_weights_xavier_uniform_(self)

class Model(nn.Module):

    def __init__(self, expert: M.Expert, shared_vae: M.VAE):
        super().__init__()
        
        self.expert = expert
        self.shared_vae = shared_vae

    def forward(self, train_input: torch.Tensor):
        
        shared_input = self.expert.encoder(train_input)
        shared_output, mu, var, shared_encoder_outputs, shared_decoder_outputs = self.shared_vae(shared_input)
        expert_output = self.expert.decoder(shared_output)
        return shared_input, shared_output, mu, var, shared_encoder_outputs, shared_decoder_outputs, expert_output

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x, None

class SharedDecoder(nn.Module):
    def __init__(self):
        super(SharedDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 768)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x, None
    
def configure_model() -> Model:
        return Model(
            HumanExpert(
                nn.Sequential(
                    nn.Linear(60664, 1024),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.2),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                ),
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.2),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 60664),
                    nn.LeakyReLU()
                )
            ),
            SharedVAE(
                SharedEncoder(),
                SharedDecoder(),
                nn.Linear(256, 128),
                nn.Linear(256, 128)
            )
        )
     