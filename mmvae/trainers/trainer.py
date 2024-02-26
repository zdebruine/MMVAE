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
        
        self.save_every = save_every if save_every else None
        
        if log_dir is not None:
            self.writer = tb.SummaryWriter(log_dir=log_dir)
        
        self.configure_dataloader()
        self.model = self.configure_model()
        self.optimizers = self.configure_optimizers()
        self.schedulers = self.configure_schedulers()

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
        if self.__initialized and __name in  ('model', 'optimizers', 'dataloader'): 
            raise RuntimeError(f"Attribute: {__name} cannot be set after initialization")
        super().__setattr__(__name, __value)
        
    def train(self, epochs: int, load_snapshot=False):
        start_epoch = 0
        if load_snapshot and self.save_every and self.snapshot_path and self.snapshot_path != '':
            model, start_epoch = self.load_snapshot()
            self.model.load_state_dict(model)
            
        for epoch in range(start_epoch, epochs):
            self.train_epoch(epoch)
            if self.save_every is not None and (epoch + 1) % self.save_every == 0:
                self.save_snapshot(self.model)
        
        if self.writer is not None:
            self.writer.close()

class HPBaseTrainer(BaseTrainer):
    
    __required_hparams = {
        'snapshot_path': str,
        'snapshot_save_every': int,
        'tensorboard_directory': str,
        'tensorboard_run_name': str,
    }
    hparams = {}
    metrics = {}
    
    def __init__(self, device: torch.device, hparams: dict, required_hparams: dict):
        self.__required_hparams.update(required_hparams)
        self._flatten_hparams(hparams)
        self._validate_hparams()
        
        log_dir=f"{self.hparams['tensorboard_directory']}{self.hparams['tensorboard_run_name']}"
        snapshot_path=self.hparams['snapshot_path']
        save_every=self.hparams['snapshot_save_every']
        
        super(HPBaseTrainer, self).__init__(
            device,
            log_dir=None if log_dir == "" else log_dir,
            snapshot_path=None if snapshot_path == "" else snapshot_path,
            save_every=None if save_every == 0 else save_every)
    
    def _flatten_hparams(self, hparams: dict, parent = ""):
        for key, value in hparams.items():
            flattened_key = key if parent == "" else f"{parent}_{key}"
            if isinstance(value, dict):
                self._flatten_hparams(value, flattened_key)
            else:
                if value == None:
                    value = self.__required_hparams[flattened_key]() if flattened_key in self.__required_hparams else ''
                self.hparams[flattened_key] = value
    
    def _validate_hparams(self):
        for _req_key, _req_type in self.__required_hparams.items():
            if _req_key not in self.hparams:
                raise ValueError(f"Required hparam for HPBaseTrainer {_req_key} not in supplied hparams!")
            _hp = self.hparams[_req_key]
            if _hp is not None and not isinstance(_hp, _req_type):
                raise TypeError(f"Required hparam for HPBaseTrainer {_req_key} expected type {_req_type} but received {type(_hp)}")