from .cli import SCIMLCli

import sciml.utils.logging as logging

if logging.DEBUG:
    logging.debug()

__all__ = [
    'SCIMLCli',
    'logging',
]