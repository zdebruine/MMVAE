from typing import Union
import lightning as L
from lightning.pytorch.trainer.states import TrainerFn

from ._cellxgene_manager import SpeciesManager, MultiSpeciesManager
    

class SingleSpeciesDataModule(SpeciesManager, L.LightningDataModule):
    
    def __init__(
        self,
        num_workers: int,
        n_val_workers: int = None,
        n_test_workers: int = None,
        n_predict_workers: int = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Save hyperparameters
        self.save_hyperparameters(logger=True)
        self.num_workers = num_workers
        self.n_val_workers = n_val_workers if n_val_workers else num_workers
        self.n_test_workers = n_test_workers if n_val_workers else num_workers
        self.n_predict_workers = n_predict_workers if n_val_workers else num_workers
        
    def setup(self, stage):
    
        if stage in (TrainerFn.FITTING,):
            self._train_datapipe = self.train_datapipe()
            self._val_datapipe = self.val_datapipe()
        elif stage in (TrainerFn.VALIDATING,):
            self._val_datapipe = self.val_datapipe()
        elif stage in (TrainerFn.PREDICTING, TrainerFn.TESTING):
            self._test_datapipe = self.test_datapipe()
        
    def train_dataloader(self):
        dp = self._train_datapipe
        return self.create_dataloader(dp, pin_memory=self.return_dense, num_workers=self.num_workers)
    
    def val_dataloader(self):
        dp = self._val_datapipe
        return self.create_dataloader(dp, pin_memory=self.return_dense, num_workers=self.n_val_workers)
    
    def test_dataloader(self):
        dp = self._test_datapipe
        return self.create_dataloader(dp, pin_memory=self.return_dense, num_workers=self.n_test_workers)
    
    def predict_dataloader(self):
        dp = self._test_datapipe
        return self.create_dataloader(dp, pin_memory=self.return_dense, num_workers=self.n_test_workers)
    
class MultiSpeciesDataModule(MultiSpeciesManager,  L.LightningDataModule):
    
    def __init__(
        self,
        species: list[SpeciesManager],
        num_workers: int,
        n_val_workers: int = None,
        n_test_workers: int = None,
        n_predict_workers: int = None,
        **kwargs
    ):  
        super().__init__(*species, **kwargs)
        self.save_hyperparameters(logger=True)
        
        self.num_workers = num_workers
        self.n_val_workers = n_val_workers if n_val_workers else num_workers
        self.n_test_workers = n_test_workers if n_val_workers else num_workers
        self.n_predict_workers = n_predict_workers if n_val_workers else num_workers

    def setup(self, stage):
        if stage in (TrainerFn.FITTING,):
            self._train_datapipe = self.train_datapipe()
            self._val_datapipe = self.val_datapipe()
        elif stage in (TrainerFn.VALIDATING,):
            self._val_datapipe = self.val_datapipe()
        elif stage in (TrainerFn.PREDICTING, TrainerFn.TESTING):
            self._test_datapipe = self.test_datapipe()
    
    @property
    def can_pin_memory(self):
        return all(species.return_dense for species in self.species)
    
    def train_dataloader(self):
        dp = self.train_datapipe()
        return self.create_dataloader(dp, pin_memory=self.can_pin_memory, num_workers=self.num_workers)
    
    def val_dataloader(self):
        dp = self.val_datapipe()
        return self.create_dataloader(dp, pin_memory=self.can_pin_memory, num_workers=self.n_val_workers)
    
    def test_dataloader(self):
        dp = self.test_datapipe()
        return self.create_dataloader(dp, pin_memory=self.can_pin_memory, num_workers=self.n_test_workers)