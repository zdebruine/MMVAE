import random
import numpy as np
import torch
from typing import Any, Literal, Sequence, Union
import lightning as L

from ._cellxgene_manager import (
    CellxgeneManager, 
    OBS_COL_NAMES, 
    OBS_QUERY_VALUE_FILTER, 
    DEFAULT_WEIGHTS
)

from sciml.utils.constants import REGISTRY_KEYS as RK, ModelInputs

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
        self.cellx_manager = CellxgeneManager(
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
                data = self.cellx_manager.experiment_datapipe.obs_encoders[key].inverse_transform(batch_labels[:, i])
                metadata.append(data)
            metadata = np.stack(metadata, axis=1)
        
        return batch[0], metadata
    