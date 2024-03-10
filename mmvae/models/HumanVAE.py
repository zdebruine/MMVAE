import torch
import torch.nn as nn
import mmvae.models as M

class Model(nn.Module):

    def __init__(self, expert: M.Expert, shared_vae: M.VAE):
        super().__init__()
        
        self.expert = expert
        self.shared_vae = shared_vae

    def forward(self, x: torch.Tensor):
        x = self.expert.encoder(x)
        x, mu, logvar = self.shared_vae(x)
        x = self.expert.decoder(x)
        return x, mu, logvar
    
def configure_model(hparams) -> Model:
    
    def build_seq(_hparams: dict, name = None):
        """
        Build sequence of layers and activation functions where: 
        { 
            layer_key: {
                input: int,
                output: int,
                activation: 'relu' | 'leakyrelu'
                [ if 'leakyrelu' negative_slope: float ]
            }
        } 
        """
        _hparams = _hparams if name == None else { name: _hparams }
        sequence = ()
        for layer in _hparams.keys():
            input_dim, output_dim = _hparams[layer]['input'], _hparams[layer]['output']
            activation = None if 'activation' not in _hparams[layer] else _hparams[layer]['activation']
            sequence = (*sequence, nn.Linear(input_dim, output_dim))
            if activation == 'leakyrelu':
                negative_slope = 0.01 if not 'negative_slope' in layer else _hparams[layer]['negative_slope']
                sequence = (*sequence, nn.LeakyReLU(negative_slope))
            if activation == 'relu':
                sequence = (*sequence, nn.ReLU())
                
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

        
