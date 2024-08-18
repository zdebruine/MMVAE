"""
.. include:: README.md
"""

from cmmvae.runners.cli import main as cli
from cmmvae.runners.generate_umap import main as generate_umap
from cmmvae.runners.merge_predictions import main as merge_predictions
from cmmvae.runners.submit import main as submit
from cmmvae.runners.logger import main as logger

__all__ = [
    "cli",
    "generate_umap",
    "merge_predictions",
    "submit",
    "logger",
]