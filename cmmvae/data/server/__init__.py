"""
    This module contains a LightningDataModule class that manages
    Datapipes and Dataloader creation for data from cellxgene-census.

    Submodules:
        - SpeciesManager: Manages species Datapipe creation.
        - SpeciesDataModule: LightingDataModule for local npz, pkl dataset.
"""
from .cellxgene_datamodule import CellxgeneDataModule
from .anndata_dataset import AnnDataDataset, collate_fn

__all__ = [
    "CellxgeneDataModule",
]