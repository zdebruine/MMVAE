from typing import Union
from ._cellxgene_datapipe import CellxgeneDataPipe
from torch.utils.data import DataLoader

DEFAULT_WEIGHTS = dict((("train", 0.8), ("val", 0.1), ("test", 0.1)))

class CellxgeneManager:
    
    def __init__(
        self,
        directory_path: str,
        npz_masks: Union[str, list[str]],
        metadata_masks: Union[str, list[str]],
        batch_size: int = 128,
        seed: int = 42,
        split_weights: dict[str, Union[int, float]] = DEFAULT_WEIGHTS,
        return_dense: bool = False,
        verbose: bool = False,
    ):
        self.directory_path = directory_path
        self.npz_masks = npz_masks
        self.metadata_masks = metadata_masks
        self.batch_size = batch_size
        self.return_dense = return_dense
        self.split_weights = split_weights
        self.seed = seed
        self.verbose = verbose
        
    def setup(self, stage = None):
        
        self.datapipe = CellxgeneDataPipe(
            directory_path=self.directory_path,
            npz_mask=self.npz_masks,
            metadata_mask=self.metadata_masks,
            batch_size=self.batch_size,
            verbose=self.verbose,
            return_dense=self.return_dense
        )
        
        self.train, self.val, self.test = self.datapipe.random_split(
            total_length=int(3002880 / self.batch_size),
            seed=self.seed,
            weights=self.split_weights
        )
    
    def create_dataloader(self, dp, **kwargs):
        return DataLoader(
            dataset=dp, 
            batch_size=None,
            timeout=30,
            shuffle=False,
            collate_fn=lambda x: x,
            pin_memory=True,
            **kwargs)
        