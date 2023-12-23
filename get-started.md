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
# sample_shape = dataset[ 0 ].shape
# assert len( dataset ) == 89160
# assert sample_shape[ 0 ] == batch_size
# assert sample_shape[ 1 ] == 60664
```
Notice that because batch size is defined at the dataset level, a dataloader is not required. 

## Building Models
Before building an MMVAE, let's define some hyperparameters for the model:
```Python
input_size = 60664
hidden_size = 
```
The first step in building an MMVAE is to define the components of the shared/internal VAE.


```Python
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
```