
from typing import Union
from .cellxgene_datapipe import SpeciesDataPipe


class SpeciesManager:
    
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
            return tensor, metadata, self.name
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
            transform_fn=self.transform_fn())
    
    def val_datapipe(self):
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.val_npz_masks,
            metadata_masks=self.val_metadata_masks,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=self.verbose,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn())
        
    def test_datapipe(self):
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.test_npz_masks,
            metadata_masks=self.test_metadata_masks,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=False,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn())
    
    def predict_datapipe(self):
        return SpeciesDataPipe(
            directory_path=self.directory_path,
            npz_masks=self.test_npz_masks,
            metadata_masks=self.test_metadata_masks,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=False,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn())