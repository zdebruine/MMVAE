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