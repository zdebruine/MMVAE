import random
import numpy as np
import torch
from typing import Any, Literal, Sequence, Union
import lightning as L
import tiledbsoma as soma
import cellxgene_census as cell_census
import cellxgene_census.experimental.ml as census_ml

from sciml.utils.constants import REGISTRY_KEYS as RK

DEFAULT_WEIGHTS = dict((("train", 0.8), ("val", 0.1), ("test", 0.1)))

OBS_COL_NAMES = (
    "dataset_id",
    "assay", 
    "donor_id",
)

OBS_QUERY_VALUE_FILTER = 'is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]'


class CellxgeneDataManager:
    
    def __init__(
        self,
        batch_size: int,
        seed: int,
        split_weights = DEFAULT_WEIGHTS,
        obs_query_value_filter: str = OBS_QUERY_VALUE_FILTER,
        obs_column_names: tuple[str] = OBS_COL_NAMES,
        soma_chunk_size: int = None
    ):
        self.batch_size = batch_size
        self.obs_query_value_filter = obs_query_value_filter
        self.seed = seed
        self.split_weights = split_weights
        self.obs_column_names = obs_column_names
        self.soma_chunk_size = soma_chunk_size
        self.census = None
        
    def setup(self):
        self.census = cell_census.open_soma(census_version="2023-12-15")
        
        self.experiment_datapipe = census_ml.ExperimentDataPipe(
            experiment=self.census["census_data"]["homo_sapiens"],
            measurement_name="RNA",
            X_name="normalized",
            obs_query=soma.AxisQuery(value_filter=self.obs_query_value_filter),
            obs_column_names=self.obs_column_names,
            shuffle=True,
            batch_size=self.batch_size,
            seed=self.seed,
            soma_chunk_size=self.soma_chunk_size,
            use_eager_fetch=False)
        
        datapipes = self.experiment_datapipe.random_split(
            seed=self.seed,
            weights=self.split_weights)
        
        self.datapipes = dict((k, v) for k, v in zip(self.split_weights.keys(), datapipes))
    
    def teardown(self):
        if self.census and hasattr(self.census, 'close'):
            self.census.close()
        
    def create_dataloader(self, target: str, num_workers: int):
        if not target in self.datapipes:
            raise ValueError(f"target {target} not in {self.split_weights.keys()}")
        dp = self.datapipes[target]
        
        return census_ml.experiment_dataloader(
            dp,
            pin_memory=True,
            num_workers=num_workers,
            # causes OOM error
            # persistent_workers=self.trainer.training and self.hparams.num_workers > 0,
            prefetch_factor=1)
    
    
class CellxgeneDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        obs_query_value_filter: str = OBS_QUERY_VALUE_FILTER,
        obs_column_names: Sequence[str] = OBS_COL_NAMES,
        split_weights: dict[str, float] = DEFAULT_WEIGHTS,
        soma_chunk_size: int = None,
        num_workers: int = 3
    ):
        super(CellxgeneDataModule, self).__init__()
        self.cellx_manager = CellxgeneDataManager(
            batch_size, seed, split_weights, obs_query_value_filter,
            obs_column_names, soma_chunk_size
        )
        self.save_hyperparameters(logger=True)
        
    def setup(self, stage):
        self.cellx_manager.setup()
        
    def teardown(self, stage):
        self.cellx_manager.teardown()
        
    def train_dataloader(self):
        return self.cellx_manager.create_dataloader('train', self.hparams.num_workers)
    
    def val_dataloader(self):
        return self.cellx_manager.create_dataloader('val', 2)
        
    def test_dataloader(self):
        return self.cellx_manager.create_dataloader('test', 2)
        
    def predict_dataloader(self) -> Any:
        return self.cellx_manager.create_dataloader('test', 2)
    
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
    