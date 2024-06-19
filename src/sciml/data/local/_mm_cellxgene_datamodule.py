import numpy as np
import torch
from typing import Any, Literal, Sequence, Union
import lightning as L
from torch.utils.data import DataLoader

from ._cellxgene_datamodule import CellxgeneDataModule
from ._multi_modal_loader import MMLoader


from sciml.utils.constants import REGISTRY_KEYS as RK

DEFAULT_WEIGHTS = dict((("train", 0.8), ("val", 0.1), ("test", 0.1)))

class MMCellxgeneDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        human_datamodule: CellxgeneDataModule,
        mouse_datamodule: CellxgeneDataModule
    ):
        self.human_datamodule = human_datamodule
        self.mouse_datamodule = mouse_datamodule
        super().__init__()

    def setup(self, stage = None):
        self.human_datamodule.setup(stage)
        self.mouse_datamodule.setup(stage)
            
    def train_dataloader(self, **kwargs):
        human_dl = self.human_datamodule.train_dataloader(**kwargs)
        mouse_dl = self.mouse_datamodule.train_dataloader(**kwargs)
        return MMLoader(human_dl=human_dl, mouse_dl=mouse_dl)
    
    def val_dataloader(self, **kwargs):
        human_dl = self.human_datamodule.val_dataloader(**kwargs)
        mouse_dl = self.mouse_datamodule.val_dataloader(**kwargs)
        return MMLoader(human_dl=human_dl, mouse_dl=mouse_dl)
    
    def test_dataloader(self, **kwargs):
        human_dl = self.human_datamodule.test_dataloader(**kwargs)
        mouse_dl = self.mouse_datamodule.test_dataloader(**kwargs)
        return MMLoader(human_dl=human_dl, mouse_dl=mouse_dl)
    
    def predict_dataloader(self, **kwargs):
        human_dl = self.human_datamodule.predict_dataloader(**kwargs)
        mouse_dl = self.mouse_datamodule.predict_dataloader(**kwargs)
        return MMLoader(human_dl=human_dl, mouse_dl=mouse_dl)