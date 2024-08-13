"""
    This module contains a LightningDataModule and a SpeceisManager class that manages
    Datapipes and Dataloader creation.

    Submodules:
        - SpeciesManager: Manages species Datapipe creation.
        - SpeciesDataModule: LightingDataModule for local npz, pkl dataset.
.. include:: ../../../configs/data/local.yaml
"""

from cmmvae.data.local.cellxgene_datamodule import SpeciesDataModule
from cmmvae.data.local.cellxgene_manager import SpeciesManager

__all__ = [
    "SpeciesManager",
    "SpeciesDataModule",
]