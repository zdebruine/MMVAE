# Dataset
from torch.utils.data import DataLoader
from d_mmvae.Dataset import CellxGeneDataset
batch_size = 32
dataset = CellxGeneDataset( batch_size )



# Models
from torch.nn import Linear, LeakyReLU, Sequential, Sigmoid
from d_mmvae.Models import Expert, MMVAE, MultiClassDiscriminator
encoder = Sequential(
    Linear( 60664, 512 ),
    LeakyReLU()
)
mean_layer = Linear( 512, 2 )
logvar_layer = Linear( 512, 2 )
decoder = Sequential(
    Linear( 2, 512 ),
    LeakyReLU(),
    Linear( 512, 60664 )
)
discriminator = Sequential(
    Linear( 60664, 512 ),
    LeakyReLU(),
    Linear( 512, 1 ),
    Sigmoid()    # Sigmoid on single value is binary classification
)
mc_discriminator = MultiClassDiscriminator(
    layers=discriminator,
    classes=[ "a", "b", "c" ]
)



# Experts
from copy import deepcopy
experts = [
    Expert( *[
        deepcopy( arg )
        for arg in [ encoder, decoder, discriminator, name ]
    ] )
    for name in [ "a", "b", "c" ]
]



# MMVAE
from d_mmvae.Models import MMVAE
mmvae = MMVAE(
    encoder, decoder,
    mean_layer, logvar_layer,
    experts, mc_discriminator
)