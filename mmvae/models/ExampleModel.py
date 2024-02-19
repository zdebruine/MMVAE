import torch
import torch.nn as nn
import torch.nn.functional as F
import mmvae.models as M

class Model(nn.Module):

    def __init__(self, experts: nn.ModuleDict, shared_vae: M.VAE):
        super().__init__()
        for expert in experts.values():
            assert isinstance(expert, M.Expert) 
        self.experts = experts
        self.shared_vae = shared_vae

    def set_expert(self, expert: str):
        self.__expert = expert

    def forward(self, train_input: torch.Tensor):
        expert = self.experts[self.__expert]
        shared_input = expert.encoder(train_input)
        shared_output, mu, var, shared_encoder_outputs, shared_decoder_outputs = self.shared_vae(shared_input)
        expert_output = expert.decoder(shared_output)
        return shared_input, shared_output, mu, var, shared_encoder_outputs, shared_decoder_outputs, expert_output
    
class SharedEncoder(nn.Module):
    def __init__(self, num_experts):
        super(SharedEncoder, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        discriminator = self.discriminator(x)
        x = F.relu(self.fc2(x))
        return x, discriminator

class SharedDecoder(nn.Module):
    def __init__(self):
        super(SharedDecoder, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x, None
    
def configure_model(num_experts) -> Model:
        return Model(
            nn.ModuleDict({
                'expert1': M.Expert(
                    nn.Sequential(
                            nn.Linear(60664, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 60664),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Linear(60664, 256),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    )),
                'expert2': M.Expert(
                    nn.Sequential(
                            nn.Linear(60664, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 60664),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Linear(60664, 256),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    ))
            }),
            M.VAE(
                SharedEncoder(num_experts),
                SharedDecoder(),
                nn.Linear(64, 64),
                nn.Linear(64, 64)
            )
        )
     