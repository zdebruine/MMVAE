import torch
import torch.nn as nn
import torchdata.dataloader2 as dl
import inspect
import torch.utils.tensorboard as tb
from typing import Any

class BaseTrainer:

    __initialized = False

    def __init__(
        self, 
        device: str,
        log_dir: str = None,
        snapshot_path: str = None, 
        save_every: int = None
    ) -> None:
        
        self.device = device
        self.snapshot_path = snapshot_path
        self.save_every = save_every
        
        self.dataloader = self.configure_dataloader()
        self.model = self.configure_model()
        self.optimizers = self.configure_optimizers()
        self.schedulers = self.configure_schedulers()
        
        if log_dir is not None:
            self.writer = tb.SummaryWriter(log_dir=log_dir)

        self.__initialized = True

    def configure_dataloader(self) -> dl.DataLoader2:
        raise NotImplementedError()

    def configure_model(self) -> nn.Module:
        raise NotImplementedError()
    
    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        import warnings
        warnings.showwarning('Warning! - No optimizers configured during intialization', category=Warning, filename= 'trainer.py', lineno=inspect.getframeinfo(inspect.currentframe().f_back).lineno)
        return {}
    
    def configure_schedulers(self) -> dict[str, torch.optim.lr_scheduler.LRScheduler]:
        import warnings
        warnings.showwarning(message='Warning! - No schedulers configured during intialization', category=Warning, filename= 'trainer.py', lineno=45)
        return {}

    def load_snapshot(self) -> tuple[nn.Module, int]:
        # TODO: FIX
        snapshot = torch.load(self.snapshot_path)
        return snapshot["MODEL_STATE"], snapshot["EPOCHS_RUN"]
    
    def save_snapshot(self, model: nn.Module, epoch: int) -> None:
        # TODO: FIX
        snapshot = {}
        snapshot["MODEL_STATE"] = model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train_epoch(self, epoch: int):
        raise NotImplementedError("Override this method to implement training loop for one epoch")
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if self.__initialized and __name in  ('epoch', 'model', 'optimizers', 'dataloader'): 
            raise RuntimeError(f"Attribute: {__name} cannot be set after initialization")
        super().__setattr__(__name, __value)
        
    def train(self, epochs: int, load_snapshot=False):
        if load_snapshot and self.save_every and self.snapshot_path:
            # TODO: Handle snapshot loading
            pass
            
        for epoch in range(epochs):
            self.train_epoch(epoch)
            if self.save_every is not None and (epoch + 1) % self.save_every == 0:
                self.save_snapshot(self.model)
        
        if self.writer is not None:
            self.writer.close()