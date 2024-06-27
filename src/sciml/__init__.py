import os
import cli
import data
import models
import modules

import sciml.utils.logging as logging


DEBUG = os.getenv("SCIML_DEBUG")

if DEBUG:
    logging.debug()

__all__ = [
    
]