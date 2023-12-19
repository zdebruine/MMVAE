from setuptools import find_packages, setup


"""
Script for setting up this Git repository as a Pip package. The 
package can be installed at any time from the repository using one 
of the following commands:
    - pip install git+https://github.com/zdebruine/D-MMVAE.git#egg=d-mmvae --extra-index-url https://download.pytorch.org/whl/cu118
    - pip install git+ssh://git@github.com/zdebruine/D-MMVAE.git#egg=d-mmvae --extra-index-url https://download.pytorch.org/whl/cu118

NOTE: A Python version >=3.7 ,<3.10 is required to satisfy 
torch==2.1.1+cu118. Pytorch only seems to support CUDA 11.8, 12.1 
toolkits; HPC provides modules for CUDA 11.8, 12.0 toolkits, but 
not for 12.1 toolkit. 2.1.1 is still the latest stable release 
of PyTorch.

This Pip skeleton was created by following Michael Kim's Pip 
package tutorial:
    - https://github.com/MichaelKim0407/tutorial-pip-package
"""


# TODO:
#   - Create an account on PyPi.
#   - Package library for distribution.
#     ( python setup.py sdist )
#   - Upload the resulting .TAR.GZ file to PyPi.
#     ( twine upload dist/<package>.tar.gz )


setup(
    name="D-MMVAE",
    description="A research project on diagonal mixture-of-experts variational autoencoding (D-MMVAE).",
    version="0.1.0.dev2",
    url="https://github.com/zdebruine/D-MMVAE",
    author="GVSU Applied Computing Institute",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "torch==2.1.2+cu118"
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov"
        ]
    }
)
