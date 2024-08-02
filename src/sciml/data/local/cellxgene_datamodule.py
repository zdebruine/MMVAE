from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from .cellxgene_datapipe import SpeciesDataPipe
from .cellxgene_manager import SpeciesManager
from .multi_modal_loader import MultiModalDataLoader
    


class SpeciesDataModule(LightningDataModule):
    
    def __init__(
        self,
        species: list[SpeciesManager],
        num_workers: int,
        n_val_workers: int = None,
        n_test_workers: int = None,
        n_predict_workers: int = None,
    ):  
        super().__init__()
        self.save_hyperparameters(logger=True)
        
        self.species = species
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
    
    def train_datapipe(self):
        return [species.train_datapipe() for species in self.species]
    
    def val_datapipe(self):
        return [species.val_datapipe() for species in self.species]
    
    def test_datapipe(self):
        return [species.test_datapipe() for species in self.species]
    
    @property
    def can_pin_memory(self):
        return all(species.return_dnese for species in self.species)
    
    def train_dataloader(self):
        dps = list(self.train_datapipe())
        return self.create_dataloader(*dps, pin_memory=self.can_pin_memory, num_workers=self.num_workers)
    
    def val_dataloader(self):
        dps = list(self.val_datapipe())
        return self.create_dataloader(*dps, pin_memory=self.can_pin_memory, num_workers=self.n_val_workers)
    
    def test_dataloader(self):
        dps = list(self.test_datapipe())
        return self.create_dataloader(*dps, pin_memory=self.can_pin_memory, num_workers=self.n_test_workers)
    
    def predict_dataloader(self):
        dps = list(self.test_datapipe())
        return self.create_dataloader(*dps, pin_memory=self.can_pin_memory, num_workers=self.n_test_workers)
    
    def create_dataloader(self, *species: SpeciesDataPipe, **kwargs):
        dataloaders = [
            DataLoader(
                dataset=dp, 
                batch_size=None,
                shuffle=False,
                collate_fn=lambda x: x,
                persistent_workers=False,
                **kwargs)
            for dp in species
        ]
        if len(dataloaders) == 1:
            return dataloaders[0]
        return MultiModalDataLoader(*dataloaders)
