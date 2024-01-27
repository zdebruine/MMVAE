# Get Started

## Prerequisites
When developing remotely on Clipper, it is recommended to avoid creating extra copies of PyTorch and other large project dependencies to conserve storage on the cluster. This can be accomplished by loading the latest version of the `ml-python` TCL module:
```
module load ml-python
```
If working with GPUs, loading one of the CUDA Toolkit modules is required:
```
module load cuda< 11.8 | 12.0 >/toolkit
```

## Installation
The repository can be installed with Pip by using one of the following commands:
```
pip install git+https://github.com/zdebruine/D-MMVAE.git#egg=d-mmvae \
--extra-index-url https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
```
```
pip install git+ssh://git@github.com/zdebruine/D-MMVAE.git#egg=d-mmvae \
--extra-index-url https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
```
NOTE: Python version >=3.7 ,<3.10 is required to satisfy the PyTorch dependency.

## Module Structure

# data
 - Dataloaders built for D-MMVAE leverage pytorch IterDataPipe pipeline. Due to multi-process loading requirements each modality will have it's own dataloader.
 - To create a Dataloader create a pipline in the data.pipes module and use torchdata.dataloader2.Dataloader2.
 - All generic functional pipes are registered in data.pipes.utils.
 - MultiModalLoader is designed to take in dataloaders as args and stochasticly draw samples from provided loaders.

# models
 - Models.py contains generic nn.Modules
 - New model architectures are contained within their own files 

# trainers
 - New trainers should inherit from BaseTrainer and are provided convienent's when it comes to saving model state and training.
 - Classes inherited from BaseTrainer need only to impelemnt the train_epoch() method.
 - Each trainer should be its own file ideally named in convention with it's associated model class and file
```python
```