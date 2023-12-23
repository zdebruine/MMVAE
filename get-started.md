# Get Started

## Prerequisites
When developing remotely on Clipper, it is recommended to avoid creating extra copies of PyTorch and other large project dependencies to conserve storage on the cluster. For this project, this can be accomplished by loading the latest version of the `ml-python` TCL module:
```
module load ml-python
```
If working with GPUs, loading one of the CUDA Toolkit modules is required:
```
module load cuda< 11.8 | 12.0 >/toolkit
```

## Installation
This repository can be installed with Pip by using one of the following commands:
```
pip install git+https://github.com/zdebruine/D-MMVAE.git#egg=d-mmvae \
--extra-index-url https://download.pytorch.org/whl/cu118
```
```
pip install git+ssh://git@github.com/zdebruine/D-MMVAE.git#egg=d-mmvae \
--extra-index-url https://download.pytorch.org/whl/cu118
```
NOTE: Python version >=3.7 ,<3.10 is required to satisfy the PyTorch dependency.

## Dataset
Creating a new dataset of cell data is simple; to initialize the dataset:
```Python
from torch.utils.data import DataLoader
from d_mmvae.Dataset import CellxGeneDataset
dataset = CellxGeneDataset( batch_size )
```
Notice that because batch size is defined at the dataset level, a dataloader is not required. 

## Building Models
The first step in building an MMVAE is to define the components of the shared/internal VAE:
```Python
from torch.nn import Linear, LeakyReLU, Sequential, Sigmoid
from d_mmvae.Models import Expert, MultiClassDiscriminator
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
    discriminator,
    classes=[ "a", "b", "c" ]
)
```

Next, we need to define a list of experts that will feed into the shared VAE. We also need to define discriminators to provide adversarial feedback to each of the experts.

Since the experts are also VAEs, they are built in the same way as the shared/internal VAE. As such, you can use copies of the shared VAE components if you need to create experts quickly for testing, debugging, etc:
```Python
from copy import deepcopy
experts = [
    Expert( *[
        deepcopy( arg )
        for arg in [ encoder, decoder, discriminator, name ]
    ] )
    for name in [ "a", "b", "c" ]
]
```

Finally, we can compile all of these pieces into a single MMVAE:
```Python
from d_mmvae.Models import MMVAE
mmvae = MMVAE(
    encoder, decoder,
    mean_layer, logvar_layer,
    experts, mc_discriminator
)
```