import random
import numpy as np
import torch
from typing import Any, Literal, Sequence, Union
import lightning as L
from torch.utils.data import DataLoader

from ._cellxgene_manager import CellxgeneManager


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
        self.cellx_manager = CellxgeneManager(
            directory_path,
            npz_masks,
            metadata_masks,
            batch_size,
            seed,
            split_weights,
            return_dense,
            verbose,
        )
        
    def setup(self, stage):
        self.cellx_manager.setup()
        
    def train_dataloader(self):
        return self.cellx_manager.create_dataloader(self.train, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return self.cellx_manager.create_dataloader(self.val, num_workers=1)
        
    def test_dataloader(self):
        return self.cellx_manager.create_dataloader(self.test, num_workers=1)
        
    def predict_dataloader(self) -> Any:
        return self.cellx_manager.create_dataloader(self.test)
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        
        return {
            RK.X: batch[0],
            RK.METADATA: batch[1],
        }
        
def collate_fn(data):
    return data