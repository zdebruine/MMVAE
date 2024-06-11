import os
from sciml.models import VAEModel
from sciml.data import CellxgeneDataModule

import sciml.utils.logging as logging


DEBUG = os.getenv("SCIML_DEBUG")

if DEBUG:
    logging.debug()
    

__all__ = [
    "CellxgeneDataModule",
    "VAEModel",
]