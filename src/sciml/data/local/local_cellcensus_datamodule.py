import random
import numpy as np
import torch
from typing import Any, Literal, Sequence, Union
import lightning as L
from torch.utils.data import DataLoader

from .local_cellcensus_datapipe import LocalCellCensusDataPipe


from sciml.utils.constants import REGISTRY_KEYS as RK

DEFAULT_WEIGHTS = dict((("train", 0.8), ("val", 0.1), ("test", 0.1)))


class CellxgeneDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        split_weights: dict[str, float] = DEFAULT_WEIGHTS,
        num_workers: int = None,
        directory_path: str = None,
        npz_masks: Union[str, list[str]] = None,
        metadata_masks: Union[str, list[str]] = None,
        verbose: bool = False,
        return_dense: bool = True,
    ):
        super(CellxgeneDataModule, self).__init__()
        self.save_hyperparameters(logger=True)
        
    def setup(self, stage):
        
        self.datapipe = LocalCellCensusDataPipe(
            directory_path=self.hparams.directory_path,
            npz_mask=self.hparams.npz_masks,
            metadata_mask=self.hparams.metadata_masks,
            batch_size=self.hparams.batch_size,
            verbose=self.hparams.verbose,
            return_dense=self.hparams.return_dense
        )
        
        self.train, self.val, self.test = self.datapipe.random_split(
            total_length=int(3002880 / self.hparams.batch_size),
            seed=self.hparams.seed,
            weights=self.hparams.split_weights
        )
    
    def create_dataloader(self, dp, **kwargs):
        return DataLoader(
            dataset=dp, 
            batch_size=None,
            timeout=30,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            **kwargs)
        
    def train_dataloader(self):
        return self.create_dataloader(self.train, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return self.create_dataloader(self.val, num_workers=1)
        
    def test_dataloader(self):
        return self.create_dataloader(self.test, num_workers=1)
        
    def predict_dataloader(self) -> Any:
        return self.create_dataloader(self.test)
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        
        return {
            RK.X: batch[0],
            RK.METADATA: batch[1],
        }
        
def collate_fn(data):
    return data