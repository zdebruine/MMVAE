from dataclasses import dataclass
from typing import Optional, Union

from .cellxgene_datapipe import SpeciesDataPipe
    
    
    
@dataclass
class LocalFileDataset:
    npz_files: Union[str, list[str]]
    metadata_files: Union[str, list[str]]
    directory: Optional[str] = None


class SpeciesManager:
    
    def __init__(
        self,
        name: str,
        train_files: LocalFileDataset,
        val_files: LocalFileDataset,
        test_files: LocalFileDataset,
        directory: Optional[str] = None,
        batch_size: int = 128,
        return_dense: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        assert directory or all(l.directory for l in (train_files, val_files, test_files)), ValueError(
            """Global directory is not set and at least one file set does not have a directory. Absolute paths not allowed"""
        )
        if directory:
            for file_set in (train_files, val_files, test_files):
                if not file_set.directory:
                    file_set.directory = directory

        self.directory = directory
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.return_dense = return_dense
        self.verbose = verbose
        self.name = name
        
    def transform_fn(self):
        def generator(source):
            tensor, metadata = source
            return tensor, metadata, self.name
        return generator
    
    def configure_datapipe(self, file_set: LocalFileDataset):
        return SpeciesDataPipe(
            directory_path=file_set.directory,
            npz_masks=file_set.npz_files,
            metadata_masks=file_set.metadata_files,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            return_dense=self.return_dense,
            transform_fn=self.transform_fn())

    def train_datapipe(self):
        return self.configure_datapipe(self.train_files)

    def val_datapipe(self):
        return self.configure_datapipe(self.val_files)

    def test_datapipe(self):
        return self.configure_datapipe(self.test_files)