from typing import Union
from ._cellxgene_datapipe import CellxgeneDataPipe
from torch.utils.data import DataLoader
import warnings

class CellxgeneManager:
    
    __setup_initialized: bool = False
    
    def __init__(
        self,
        directory_path: str,
        train_npz_masks: Union[str, list[str]],
        train_metadata_masks: Union[str, list[str]],
        val_npz_masks: Union[str, list[str]],
        val_metadata_masks: Union[str, list[str]],
        test_npz_masks: Union[str, list[str]],
        test_metadata_masks: Union[str, list[str]],
        num_workers: int = None,
        n_val_workers: int = None,
        n_test_workers: int = None,
        n_predict_workers: int = None,
        batch_size: int = 128,
        seed: int = 42,
        return_dense: bool = False,
        verbose: bool = False,
    ):
        self.directory_path = directory_path
        self.train_npz_masks = train_npz_masks
        self.train_metadata_masks = train_metadata_masks
        self.val_npz_masks = val_npz_masks
        self.val_metadata_masks = val_metadata_masks
        self.test_npz_masks = test_npz_masks
        self.test_metadata_masks = test_metadata_masks
        self.batch_size = batch_size
        self.return_dense = return_dense
        self.seed = seed
        self.verbose = verbose
        self.num_workers = num_workers
        self.n_val_workers = n_val_workers
        self.n_test_workers = n_test_workers
        self.n_predict_workers = n_predict_workers
        
    def setup(self, stage = None):
        
        _all = stage == None
        
        if _all or stage == 'train':
            self.train_dp = CellxgeneDataPipe(
                directory_path=self.directory_path,
                npz_mask=self.train_npz_masks,
                metadata_mask=self.train_metadata_masks,
                batch_size=self.batch_size,
                verbose=self.verbose,
                return_dense=self.return_dense
            )
        
        if _all or stage == 'val':
            self.val_dp = CellxgeneDataPipe(
                directory_path=self.directory_path,
                npz_mask=self.val_npz_masks,
                metadata_mask=self.val_metadata_masks,
                batch_size=self.batch_size,
                verbose=self.verbose,
                return_dense=self.return_dense
            )
            
        if _all or stage == 'test':
            self.test_dp = CellxgeneDataPipe(
                directory_path=self.directory_path,
                npz_mask=self.test_npz_masks,
                metadata_mask=self.test_metadata_masks,
                batch_size=self.batch_size,
                verbose=self.verbose,
                return_dense=self.return_dense
            )
            
        self.__setup_initialized = True
            
    def train_dataloader(self, **kwargs):
        
        if not self.__setup_initialized:
            raise RuntimeError("Please call setup before creating dataloaders")
        
        if 'num_workers' not in kwargs:
            kwargs['num_workers'] = self.num_workers
        
        return self.create_dataloader(self.train_dp, **kwargs)
    
    def val_dataloader(self, **kwargs):
        
        if not self.__setup_initialized:
            raise RuntimeError("Please call setup before creating dataloaders")
        
        if 'num_workers' not in kwargs:
            n_wrks = self.n_val_workers
            if n_wrks == None:
                n_wrks = self.num_workers
            kwargs['num_workers'] = n_wrks
        else:
            warnings.warn("param num_workers in val_dataloader overriden by kwargs supplied")
            
        return self.create_dataloader(self.val_dp, **kwargs)
    
    def test_dataloader(self, **kwargs):

        if not self.__setup_initialized:
            raise RuntimeError("Please call setup before creating dataloaders")
        
        if 'num_workers' not in kwargs:
            n_wrks = self.n_test_workers
            if n_wrks == None:
                n_wrks = self.num_workers
            kwargs['num_workers'] = n_wrks
        else:
            warnings.warn("param num_workers in test_dataloader overriden by kwargs supplied")
            
        return self.create_dataloader(self.test_dp, **kwargs)
    
    def predict_dataloader(self, **kwargs):
        
        if not self.__setup_initialized:
            raise RuntimeError("Please call setup before creating dataloaders")
        
        if 'num_workers' not in kwargs:
            n_wrks = self.n_test_workers
            if n_wrks == None:
                n_wrks = self.num_workers
            kwargs['num_workers'] = n_wrks
        else:
            warnings.warn("param num_workers in predict_dataloader overriden by kwargs supplied")
            
        return self.create_dataloader(self.test_dp, **kwargs)
    
    def create_dataloader(self, dp, **kwargs):
        return DataLoader(
            dataset=dp, 
            batch_size=None,
            timeout=30,
            shuffle=False,
            collate_fn=lambda x: x,
            pin_memory=self.return_dense,
            **kwargs)
        