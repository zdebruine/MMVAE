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
    
def configure_model(hparams) -> Model:
    
    def build_seq(_hparams: dict, name = None):
        _hparams = _hparams if name == None else {name: _hparams}
        sequence = ()
        for layer in _hparams.keys():
            input_dim, output_dim = _hparams[layer]['input'], _hparams[layer]['output']
            activation = None if 'activation' not in _hparams[layer] else _hparams[layer]['activation']
            sequence = (*sequence, nn.Linear(input_dim, output_dim))
            if activation == 'leakyrelu':
                negative_slope = 0.01 if not 'negative_slope' in layer else _hparams[layer]['negative_slope']
                activation = f"{layer}_activation"
                sequence = (*sequence, nn.LeakyReLU(negative_slope))
                
        if len(sequence) == 1:
            return sequence[0]
        return nn.Sequential(*sequence)

    
    return Model(
            M.Expert(
                build_seq(hparams["expert"]['encoder']['model']),
                build_seq(hparams["expert"]['decoder']['model']),
            ),
            M.VAE(
                build_seq(hparams["shr_vae"]['model']['encoder']),
                build_seq(hparams["shr_vae"]['model']['decoder']),
                build_seq(hparams["shr_vae"]['model']['mu'], "mu"),
                build_seq(hparams["shr_vae"]['model']['logvar'], "logvar"),
            )
        )

        
