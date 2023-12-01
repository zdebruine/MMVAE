from setuptools import find_packages, setup


"""
Script for setting up this Git repository as a Pip package. The 
package can be installed at any time from the repository using one 
of the following commands:
  - pip install git+https://github.com/<username>/D-MMVAE.git#egg=d-mmvae
  - pip install git+ssh://git@github.com/<username>/D-MMVAE.git#egg=d-mmvae

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
    version="0.1.0.dev0",
    url="https://github.com/zdebruine/D-MMVAE",
    author="GVSU Applied Computing Institute",
    packages=find_packages()
)