from d_mmvae.Models import VAE
from torch import nn

DATA_SIZE = 28*28
LATENT_SIZE = 10

# Load MNIST here

encoder = nn.Sequential(
    nn.Linear(DATA_SIZE, 256),
    nn.ReLU(),
    nn.Linear(256, LATENT_SIZE),
    nn.ReLU()
)

decoder = nn.Sequential(
    nn.Linear(LATENT_SIZE, 256),
    nn.ReLU(),
    nn.Linear(256, DATA_SIZE),
    nn.ReLU()
)

mu = nn.Linear(LATENT_SIZE, LATENT_SIZE)
var = nn.Linear(LATENT_SIZE, LATENT_SIZE) 

vae = VAE(encoder, decoder, mu, var)
discriminator = nn.Sequential(
    nn.Linear(DATA_SIZE, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
