from setuptools import find_packages, setup
import csv
import os
from setuptools.command.install import install

"""
Script for setting up this Git repository as a Pip package. The 
package can be installed at any time from the repository using one 
of the following commands:
    - pip install git+https://github.com/zdebruine/D-MMVAE.git#egg=d-mmvae --extra-index-url https://download.pytorch.org/whl/nightly/cu121/
    - pip install git+ssh://git@github.com/zdebruine/D-MMVAE.git#egg=d-mmvae --extra-index-url https://download.pytorch.org/whl/nightly/cu121/

# NOTE TO USERS: 
# This package requires the nightly build of PyTorch. The nightly build 
# can be installed separately using the following command:
# pip install torch==nightly -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html
# Please ensure to replace 'cu118' with the CUDA version that matches your system.

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


def parse_requirements(file_name):
    """Parse a requirements file into a list of packages."""
    requirements = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            package = row['package']
            version = row['version']
            if version == 'nightly':
                # Handle special cases like nightly builds
                requirements.append(f"{package}==nightly")
            else:
                requirements.append(f"{package}{version}")
    return requirements

class CustomInstallCommand(install):
    """Customized setuptools install command - installs with specific index URLs."""

    def run(self):

        # Install PyTorch, TorchVision, and TorchAudio from the nightly builds
        os.system("pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu")

        # Run the standard install process
        install.run(self)

setup(
    name="MMVAE",
    description="A research project on diagonal mixture-of-experts variational autoencoding (MMVAE).",
    version="0.1.1.dev2",
    url="https://github.com/zdebruine/MMVAE",
    author="GVSU Applied Computing Institute",
    packages=find_packages(include=['src/cmmvae', 'src/cmmvae.*']),  # Ensure cmmvae is included
    install_requires=[
        "cellxgene-census",
        "jsonargparse",
        "lightning",
        "omegaconf",
        "snakemake",
        "tensorboard_plugin_profile",
        "torch-tb-profiler",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov"
        ]
    },
    cmdclass={
        'install': CustomInstallCommand,
    }
)