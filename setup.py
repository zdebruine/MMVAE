from setuptools import setup
from hello_world import __version__



setup(
    name="D-MMVAE",
    description="A research project on diagonal mixture-of-experts variational autoencoding (D-MMVAE).",
    version=__version__,
    url="https://github.com/zdebruine/D-MMVAE",
    author="GVSU Applied Computing Institute",
    py_modules=[ "hello_world" ]
)