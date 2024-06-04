import numpy as np
from typing import Any, Sequence, Union
import lightning as L
import tiledbsoma as soma
import cellxgene_census as cell_census
import cellxgene_census.experimental.ml as census_ml
import argparse


DEFAULT_WEIGHTS = {"train": 0.8, "test": 0.15, "val": 0.05}

OBS_COL_NAMES = (
    "dataset_id",
    "assay", 
    "donor_id",
)

OBS_QUERY_VALUE_FILTER = 'is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]'


class CellxgeneDataLoaders:
    
    def __init__(
        self,
        num_workers: int = 3
    ):
        super(CellxgeneDataLoaders, self).__init__()
        self.census = None
        self.num_workers = num_workers
        
    def setup(
        self, 
        batch_size: int,
        seed: int = 42,
        obs_query_value_filter: str = OBS_QUERY_VALUE_FILTER,
        obs_column_names: str = OBS_COL_NAMES,
        soma_chunk_size: int = 1000,
        weights = DEFAULT_WEIGHTS
    ):
        self.census = cell_census.open_soma(census_version="2023-12-15")
        
        experiment_datapipe = census_ml.ExperimentDataPipe(
            experiment=self.census["census_data"]["homo_sapiens"],
            measurement_name="RNA",
            X_name="normalized",
            obs_query=soma.AxisQuery(value_filter=obs_query_value_filter),
            obs_column_names=obs_column_names,
            shuffle=True,
            batch_size=batch_size,
            seed=seed,
            soma_chunk_size=soma_chunk_size)
        
        self.obs_encoders = experiment_datapipe.obs_encoders
            
        self.train_dp, self.test_dp, self.val_dp = experiment_datapipe.random_split(
            total_length=len(experiment_datapipe),
            weights=weights, 
            seed=seed)
        
    def teardown(self, stage):
        if self.census and hasattr(self.census, 'close'):
            self.census.close()
        
    def cell_census_dataloader(self, dp):
        return census_ml.experiment_dataloader(
            dp,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
        
    def train_dataloader(self):
        return self.cell_census_dataloader(self.train_dp)
        
    def test_dataloader(self):
        return self.cell_census_dataloader(self.test_dp)
        
    def val_dataloader(self):
        return self.cell_census_dataloader(self.val_dp)
    
    def predict_dataloader(self) -> Any:
        return self.cell_census_dataloader(self.test_dp)
