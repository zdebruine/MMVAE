import torch
import torch.nn as nn
import torchdata.dataloader2 as dl
from typing import Any

class BaseTrainer:

    __initialized = False

    def __init__(
        self, 
        model: nn.Module, 
        optimizers: dict[str, torch.optim.Optimizer], 
        dataloader: dl.DataLoader2, 
        device: str,
        snapshot_path: str = None, 
        save_every: int = None
    ) -> None:
        self.model = model
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.snapshot_path = snapshot_path
        self.save_every = save_every
        self.device = device
        self.__epoch = 0
        self.__initialized = True

    def load_snapshot(self) -> tuple[int, nn.Module]:
        snapshot = torch.load(self.snapshot_path)
        return snapshot["MODEL_STATE"], snapshot["EPOCHS_RUN"]
    
    def save_snapshot(self, model: nn.Module, epoch: int) -> None:
        snapshot = {}
        snapshot["MODEL_STATE"] = model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train_epoch(self):
        raise NotImplementedError("Override this method to implement training loop for one epoch")
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if self.__initialized and __name in  ('epoch', 'model', 'optimizers', 'dataloader', 'storage'): 
            raise RuntimeError(f"Attribute: {__name} cannot be set after initialization")
        super().__setattr__(__name, __value)

    @property
    def epoch(self):
        self.__epoch
        
    def train(self, epochs: int, load_snapshot=False):
        if load_snapshot and self.save_every and self.snapshot_path:
            self.load_snapshot()
            
        for epoch in range(epochs):
            self.__epoch = epoch
            self.train_epoch()
            if (epoch + 1) % self.save_every == 0:
                self.save_snapshot(self.model)
        