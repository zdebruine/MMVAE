from typing import Union
import lightning as L

from ._cellxgene_manager import CellxgeneManager

class CellxgeneDataModule(CellxgeneManager, L.LightningDataModule):
    
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
        # Initialize CellxgeneManager with all necessary arguments
        CellxgeneManager.__init__(
            self,
            directory_path=directory_path,
            train_npz_masks=train_npz_masks,
            train_metadata_masks=train_metadata_masks,
            val_npz_masks=val_npz_masks,
            val_metadata_masks=val_metadata_masks,
            test_npz_masks=test_npz_masks,
            test_metadata_masks=test_metadata_masks,
            num_workers=num_workers,
            n_val_workers=n_val_workers,
            n_test_workers=n_test_workers,
            n_predict_workers=n_predict_workers,
            batch_size=batch_size,
            seed=seed,
            return_dense=return_dense,
            verbose=verbose,
        )
        
        # Initialize LightningDataModule
        L.LightningDataModule.__init__(self)
        
        # Save hyperparameters
        self.save_hyperparameters(logger=True)