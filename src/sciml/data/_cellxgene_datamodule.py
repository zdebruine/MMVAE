import random
import numpy as np
import torch
from typing import Any, Sequence, Union
import lightning as L
import tiledbsoma as soma
import cellxgene_census as cell_census
import cellxgene_census.experimental.ml as census_ml

from sciml._constant import REGISTRY_KEYS as RK

DEFAULT_WEIGHTS = { "train": 0.8, "val": 0.1, "test": 0.1 }

OBS_COL_NAMES = (
    "dataset_id",
    "assay", 
    "donor_id",
)

OBS_QUERY_VALUE_FILTER = 'is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]'


class CellxgeneDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        batch_size: int = 64,
        seed: int = 42,
        obs_query_value_filter: str = OBS_QUERY_VALUE_FILTER,
        obs_column_names: Sequence[str] = OBS_COL_NAMES,
        weights: dict[str, float] = DEFAULT_WEIGHTS,
        soma_chunk_size: int = None,
        num_workers: int = 3
    ):
        super(CellxgeneDataModule, self).__init__()
        self.save_hyperparameters(logger=True)
        self.census = None
        
    def setup(self, stage):
        self.census = cell_census.open_soma(census_version="2023-12-15")
        
        self.experiment_datapipe = census_ml.ExperimentDataPipe(
            experiment=self.census["census_data"]["homo_sapiens"],
            measurement_name="RNA",
            X_name="normalized",
            obs_query=soma.AxisQuery(value_filter=self.hparams.obs_query_value_filter),
            obs_column_names=self.hparams.obs_column_names,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            seed=self.hparams.seed,
            soma_chunk_size=self.hparams.soma_chunk_size)
        
        self.train, self.val, self.test = self.experiment_datapipe.random_split(
            seed=self.hparams.seed,
            weights=self.hparams.weights)
        
    def teardown(self, stage):
        if self.census and hasattr(self.census, 'close'):
            self.census.close()
        
    def cell_census_dataloader(self, dp):
        return census_ml.experiment_dataloader(
            dp,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.trainer.training and self.hparams.num_workers > 0,
            prefetch_factor=1,
        )
        
    def train_dataloader(self):
        return self.cell_census_dataloader(self.train)
    
    def val_dataloader(self):
        return self.cell_census_dataloader(self.val)
        
    def test_dataloader(self):
        return self.cell_census_dataloader(self.test)
        
    def predict_dataloader(self) -> Any:
        return self.cell_census_dataloader(self.test)
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        
        metadata = None
        if self.trainer.predicting or self.trainer.validating:
            batch_labels = batch[1]
            metadata = []
            for i, key in enumerate(self.hparams.obs_column_names, start=1):
                data = self.experiment_datapipe.obs_encoders[key].inverse_transform(batch_labels[:, i])
                metadata.append(data)
            metadata = np.stack(metadata, axis=1)
        
        return {
            RK.X: batch[0],
            RK.Y: batch[1],
            RK.METADATA: metadata,
        }
    