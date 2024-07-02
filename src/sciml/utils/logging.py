import psutil
import logging
import os

DEBUG = os.getenv("SCIML_LOGGING_STATUS", "").upper() == 'DEBUG'

def debug():
    # Get memory statistics
    mem = psutil.virtual_memory()

    # Total physical memory (in GB)
    total_memory_gb = mem.total / (1024**3)

    # Available memory (in GB)
    available_memory_gb = mem.available / (1024**3)

    print(f"Total Memory: {total_memory_gb} GB", flush=True)
    print(f"Available Memory: {available_memory_gb} GB", flush=True)

    filename = f"/mnt/projects/debruinz_project/denhofja/debug_logs/{os.getpid()}.debug"
    print("Debug Directory:", filename, flush=True)
    hdlr = logging.FileHandler(filename=filename, mode='w')
    # Specific loggers for PyTorch and PyTorch Lightning
    torch_loggers = [
        'torch.utils.data.dataloader',
        'lightning',  # PyTorch Lightning main logger
        'lightning.pytorch',  # Additional PyTorch Lightning utilities
        'lightning.fabric',  # PyTorch Lightning for different computation environments
        'cellxgene_census',
        'cellxgene_census.experimental.util',
        'cellxgene_census.experimental',
        'cellxgene_census.experimental.pytorch',
        'anndata'
    ]

    # Set DEBUG level for each specified logger to ensure all debug messages are captured
    for logger_name in torch_loggers:
        logger = logging.getLogger(logger_name)
        for handl in logger.handlers:
            logger.removeHandler(handl)
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)  # Set to debug to capture all messages for these modules