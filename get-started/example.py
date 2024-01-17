import torch
import torchdata

# Dataset
from d_mmvae.DataLoaders import CellCensusPipeLine
batch_size = 32
    
_experts = ["A", "B", "C"]

datapipes = [CellCensusPipeLine(expert, batch_size, "/active/debruinz_project/tony_boos/csr_chunks", ['chunk'], 5) for expert in _experts.keys() ]
# MultiplexerLongest will go until longest exhausted
dataloader = torchdata.datapipes.iter.Multiplexer(*datapipes) # terminates at shortest length datapipe
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, hidden_size, discriminator_hidden_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_size, discriminator_hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(discriminator_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Expert:

    def __init__(self, name, encoder: torch.nn.Module, decoder: torch.nn.Module):
        self.name = name
        self.encoder = encoder
        self.decoder = decoder

class Network(torch.nn.Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        output = []
        for layer in self.layers:
            if isinstance(layer, Discriminator):
                output.append(layer(x))
            else:
                x = layer(x)
        return x, output
    
class Encoder(Network):
    def __init__(self, *layers):
        super(Decoder, self).__init__(*layers)

class Decoder(Network):
    def __init__(self, *layers):
        super(Decoder, self).__init__(*layers)

class VAE(torch.nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, mean: nn.Linear, var: nn.Linear) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        
    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(var)
        return mean + var*eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x, encoder_outputs = self.encoder(x)
        mu = self.mean(x)
        var = self.var(x)
        z = self.reparameterize(mu, torch.exp(0.5 * var))
        x_hat, decoder_outputs = self.decoder(z)
        return x_hat, mu, var, encoder_outputs, decoder_outputs



class MMVAE(torch.nn.Module):

    def __init__(self, experts: dict[str, Expert], model: VAE):
        super(MMVAE, self).__init__()
        self.experts = experts
        self.model = model
    
    def forward(self, name: str, input: torch.Tensor):
        exp_enc_out = self.experts[name].encoder(input)
        shr_enc_out, mu, var, shr_enc_outs = self.model.encoder(exp_enc_out)
        shr_dec_out, shr_dec_outs = self.model.decoder(exp_enc_out)
        exp_dec_out  = self.experts[name].decoder(shr_dec_out)
        return exp_enc_out, exp_dec_out, shr_enc_out, shr_dec_out, mu, var, shr_enc_outs, shr_dec_outs

experts = {}
for expert in _experts:
    experts[expert] = Expert(
        expert, 
        torch.nn.Sequential(
            torch.nn.Linear(60664, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        ),
        torch.nn.Sequential(
            torch.nn.Linear(512, 60664),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        ),
    )
discriminator = Discriminator()
vae = VAE(
    Encoder(
        nn.Linear(512, 256),
        nn.ReLU(),
        discriminator,
        nn.Linear(256, 64),
        nn.ReLU(),
    ),
    Decoder(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 60664)
    )
)

mmvae = MMVAE(experts, vae)

expert_optimizer: dict[str, Expert] = {}
for expert in experts:
    expert_optimizer[expert] = [torch.optim.Adam(mmvae.experts[expert].encoder.parameters()), torch.optim.Adam(mmvae.experts[expert].decoder.parameters())]

discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
shared_optimizer = torch.optim.Adam(mmvae.parameters())

for train_input, expert in dataloader:
    mmvae.train()

    exp_enc_out, exp_dec_out, shr_enc_out, shr_dec_out, mu, var, shr_enc_outs, shr_dec_outs = mmvae(train_input)

    shared_optimizer.zero_grad()
    recon_loss = torch.nn.MSELoss()(shr_dec_out, exp_enc_out)
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    loss = recon_loss + KLD
    loss.backward()
    shared_optimizer.step()

    expert_optimizer[expert].encoder.zero_grad()
    expert_optimizer[expert].decoder.zero_grad()
    expert_recon_loss = torch.nn.MSELoss()(exp_dec_out, train_input.to_dense())
    expert_recon_loss.backward()
    expert_optimizer[expert].encoder.step()
    expert_optimizer[expert].decoder.step()

    discriminator_optimizer.zero_grad()
    torch.nn.MSELoss()(shr_enc_outs[0], 1).backward()
    discriminator_optimizer.step()




    
    


    

    



    
    
