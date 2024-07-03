from typing import Literal, Union
from ._cellxgene_datapipe import SpeciesDataPipe, MultiSpeciesDataPipe
from torch.utils.data import DataLoader
import warnings

from sciml.utils.constants import REGISTRY_KEYS as RK


class BaseSpeciesManager:
    
    def train_datapipe(self):
        raise NotImplementedError()
    
    def val_datapipe(self):
        raise NotImplementedError()
    
    def test_datapipe(self):
        raise NotImplementedError()
    
    def predict_datapipe(self):
        raise NotImplementedError()
    
    def create_dataloader(self, dp, **kwargs):
        return DataLoader(
            dataset=dp, 
            batch_size=None,
            shuffle=False,
            collate_fn=lambda x: x,
            persistent_workers=False,
            **kwargs)
        

class SpeciesManager(BaseSpeciesManager):
    
    def __init__(
        self,
        name: str,
        directory_path: str,
        train_npz_masks: Union[str, list[str]],
        train_metadata_masks: Union[str, list[str]],
        val_npz_masks: Union[str, list[str]],
        val_metadata_masks: Union[str, list[str]],
        test_npz_masks: Union[str, list[str]],
        test_metadata_masks: Union[str, list[str]],
        batch_size: int = 128,
        return_dense: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.directory_path = directory_path
        self.train_npz_masks = train_npz_masks
        self.train_metadata_masks = train_metadata_masks
        self.val_npz_masks = val_npz_masks
        self.val_metadata_masks = val_metadata_masks
        self.test_npz_masks = test_npz_masks
        self.test_metadata_masks = test_metadata_masks
        self.batch_size = batch_size
        self.return_dense = return_dense
        self.verbose = verbose
        self.name = name
        
    def transform_fn(self):
        def generator(source):
            tensor, metadata = source
            return {
                RK.X: tensor,
                RK.METADATA: metadata,
                RK.EXPERT_ID: self.name
            }
        return generator

    
    def train_datapipe(self):
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.train_npz_masks,
            metadata_masks=self.train_metadata_masks,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn()
        )
    
    def val_datapipe(self):
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.val_npz_masks,
            metadata_masks=self.val_metadata_masks,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=self.verbose,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn()
        )
        
    def test_datapipe(self):
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.test_npz_masks,
            metadata_masks=self.test_metadata_masks,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=False,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn()
        )
        
class MultiSpeciesManager(BaseSpeciesManager):
    
    def __init__(self, *species: SpeciesManager):
        super().__init__()
        self.species = species
        
    def multi_species_datapipe(self, *species: SpeciesDataPipe, select_fn='random'):
        return MultiSpeciesDataPipe(*species, selection_fn=select_fn)
    
    def train_datapipe(self):
        return (species.train_datapipe() for species in self.species)
    
    def val_datapipe(self):
        return (species.val_datapipe() for species in self.species)
    
    def test_datapipe(self):
        return (species.train_datapipe() for species in self.species)

