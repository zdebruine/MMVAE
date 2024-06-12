import os
from sciml.models import VAEModel

import sciml.utils.logging as logging


DEBUG = os.getenv("SCIML_DEBUG")

if DEBUG:
    logging.debug()