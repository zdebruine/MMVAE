from .cli import SCIMLCli

import sciml.logging as logging

if logging.DEBUG:
    logging.debug()

__all__ = [
    'SCIMLCli',
    'logging',
]