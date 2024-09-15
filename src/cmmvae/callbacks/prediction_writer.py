from typing import Any, Optional, Sequence
import os
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import numpy as np
import h5py
import torch
import warnings


def save_to_hdf5(hdf5_filepath, key, data, metadata, strict=True):
    """
    Save numpy array `data` and pandas DataFrame `metadata` to HDF5 file.
    Each column in the metadata DataFrame will be stored as a separate dataset.
    """
    with h5py.File(hdf5_filepath, "a") as h5file:
        if f"{key}/data" in h5file and f"{key}/metadata" in h5file:
            dataset = h5file[f"{key}/data"]
            new_size = dataset.shape[0] + data.shape[0]
            dataset.resize(new_size, axis=0)
            dataset[-data.shape[0] :] = data  # Append new batch

            # Append metadata column-wise
            for col in metadata.columns:
                column_data = metadata[col].values
                if f"{key}/metadata/{col}" not in h5file:
                    if strict:
                        raise RuntimeError(
                            f"metadata column {col} not in h5file for key {key}/metadata/{col}"
                        )
                    else:
                        continue

                metadata_dataset = h5file[f"{key}/metadata/{col}"]
                metadata_dataset.resize(new_size, axis=0)
                metadata_dataset[-len(column_data) :] = column_data
        else:
            # Create a new dataset for `data`
            h5file.create_dataset(
                f"{key}/data", data=data, maxshape=(None,) + data.shape[1:], chunks=True
            )

            # Create metadata datasets, one for each column
            metadata_group = h5file.create_group(f"{key}/metadata")
            for col in metadata.columns:
                metadata_group.create_dataset(
                    f"{col}", data=metadata[col].values, maxshape=(None,), chunks=True
                )


def load_from_hdf5(hdf5_filepath, key):
    """
    Load numpy array `data` and pandas DataFrame `metadata` from HDF5 file.
    """
    with h5py.File(hdf5_filepath, "r") as h5file:
        # Load the data array
        data = h5file[f"{key}/data"][:]

        # Load metadata as individual columns
        metadata_group = h5file[f"{key}/metadata"]
        metadata_dict = {col: metadata_group[col][:] for col in metadata_group.keys()}

        embedding = None
        if f"{key}/umap_embedding" in h5file:
            embedding = h5file[f"{key}/umap_embedding"][:]

        # Rebuild the DataFrame from columns
        metadata = pd.DataFrame(metadata_dict)

        return data, metadata, embedding


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        root_dir: str,
        experiment_name: str = "",
        run_name: str = "",
        hdf5_filename: str = "predictions.h5",
    ):
        super().__init__(write_interval="batch")
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.hdf5_filename = hdf5_filename
        self._curr_size = 0  # Keeps track of total rows written so far

        if os.path.exists(self.hdf5_filepath):
            warnings.warn(
                f"PredictionWriter initialized with hdf5_filepath that already exists: {self.hdf5_filepath}"
            )

    @property
    def hdf5_filepath(self):
        return os.path.join(self.save_dir, self.hdf5_filename)

    @property
    def save_dir(self):
        return os.path.join(self.root_dir, self.experiment_name, self.run_name)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        if not isinstance(prediction, dict) or not all(
            isinstance(p, (tuple, list))
            and len(p) == 2
            and (
                isinstance(p[0], (torch.Tensor, np.ndarray))
                and isinstance(p[1], pd.DataFrame)
            )
            for p in prediction.values()
        ):
            raise ValueError(
                f"Prediction must be a dictionary of type 'dict[str, tuple[torch.Tensor, pd.DataFrame]]' (got {type(prediction)})"
            )

        for key, (data, metadata) in prediction.items():
            data = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
            save_to_hdf5(
                hdf5_filepath=self.hdf5_filepath, key=key, data=data, metadata=metadata
            )

        self._curr_size += list(prediction.values())[0][0].shape[
            0
        ]  # Increment by batch size

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        super().on_predict_start(trainer, pl_module)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._curr_size = 0  # Reset after the epoch ends
        super().on_predict_epoch_end(trainer, pl_module)
