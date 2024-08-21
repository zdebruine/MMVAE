"""
.. include:: README.md
"""

from cmmvae.runners.cli import cli
from cmmvae.runners.umap_predictions import umap_predictions
from cmmvae.runners.merge_predictions import merge_predictions
from cmmvae.runners.submit import submit
from cmmvae.runners.logger import logger
from cmmvae.runners.autodoc import autodoc

__all__ = [
    "cli",
    "umap_predictions",
    "merge_predictions",
    "submit",
    "logger",
    "autodoc"
]
