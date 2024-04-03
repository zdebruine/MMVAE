from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, ReadingServiceInterface
from typing import Generator, Any
import mmvae.data.pipes as p
import torch
import random

class MultiModalLoader:
    """
    Stochastic sampler from dataloaders
    
    Args:
     - exhaust_all: exhaust all dataloaders to completion (default=True)
    """

    def __init__(self, *modals: DataLoader2, exhaust_all=True):
        self.exhaust_all = exhaust_all
        if len(modals) == 0:
            raise ValueError("A dataloader must be defined!")
        self.modals = modals
        self.__len = None

    def __len__(self):
        if self.__len == None:
            raise RuntimeError("Length of MultiModalLoader cannot be determined until one entire forward pass")
        return self.__len

    def __iter__(self) -> Generator[tuple[torch.Tensor, str, Any], Any, None]:
        loaders = [iter(modal) for modal in self.modals]
        self.__len = None
        while loaders:
            loader_idx = random.randrange(len(loaders))
            try:
                yield next(loaders[loader_idx])
                if self.__len == None:
                    self.__len = 0
                self.__len += 1
            except StopIteration as e:
                if not self.exhaust_all:
                    return
                del loaders[loader_idx]

class ChunkedCellCensusDataLoader(DataLoader2):
    """
        Dataloader wrapper for CellCensusPipeline

        Args:
         - *args: inputs to be yielded for every iteration
         - directory_path: string path to chunk location
         - masks: unix style regex matching for each string in array
         - batch_size: size of output tensor first dimension
         - num_workers: number of worker process to initialize

         Attention:
          - num_workers must be greater or equal to the total chunks to load
        """
    def __init__(self, directory_path: str = None, masks: list[str] = None, batch_size: int = None, num_workers: int = None, name=None, verbose=False): # type: ignore
        super(ChunkedCellCensusDataLoader, self).__init__(
            datapipe=p.CellCensusPipeLine(directory_path, masks, batch_size, name=name, verbose=verbose), # type: ignore
            datapipe_adapter_fn=None,
            reading_service=MultiProcessingReadingService(num_workers=num_workers)
        )
        
def configure_multichunk_dataloaders(
    train_batch_size: int,
    train_directory_path: str,
    train_masks: list[str],
    test_batch_size: int,
    test_directory_path: str,
    test_masks: list[str],
    train_num_workers: int = 2,
    test_num_workers: int = 2,
    verbose: bool =False
) -> tuple[ChunkedCellCensusDataLoader, ChunkedCellCensusDataLoader]:
    """
    Returns tuple of (train_dataset, test_dataset).
    """
        
    if train_batch_size <= 0 or test_batch_size <= 0:
        raise RuntimeError("Train and test batchsizes must be greater than 0!")
    if len(train_masks) < train_num_workers:
        train_num_workers = len(train_masks)
    if len(test_masks) < test_num_workers:
        test_num_workers = len(test_masks)
    return (
        ChunkedCellCensusDataLoader(
            directory_path=train_directory_path, 
            masks=train_masks, 
            batch_size=train_batch_size, 
            num_workers=train_num_workers,
            name='train',
            verbose=verbose
        ),
        ChunkedCellCensusDataLoader(
            directory_path=test_directory_path,
            masks=test_masks, 
            batch_size=test_batch_size, 
            num_workers=test_num_workers,
            name='test',
            verbose=verbose
        ),
    )
        
import mmvae.data.utils as utils

def configure_singlechunk_dataloader(
    data_file_path: str,
    metadata_file_path: str,
    batch_size: int,
    shuffle: bool = True,
    device: torch.device = None,
):
    """If device is not None -> the entire dataset will be loaded on device at once."""
    matrix, metadata = utils.load_data_and_metadata(data_file_path, metadata_file_path)
    
    from mmvae.data.datasets.CellCensusDataSet import CellCensusDataset, collate_fn
    if device:
        train_data = train_data.to(device)
        validation_data = validation_data.to(device)
        
    dataset = CellCensusDataset(matrix, metadata)
    
    from torch.utils.data import DataLoader
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    
    
def configure_singlechunk_dataloaders(
    data_file_path: str,
    metadata_file_path: str,
    train_ratio: float,
    batch_size: int,
    test_batch_size: int = None,
    shuffle: bool = True,
    device: torch.device = None,
):
    """
    Splits a csr_matrix provided by data_file_path with equal length metadata_file_path by train_ratio 
        which is a floating point between 0-1 (propertion of training data to test data). 
    
    If device is not None -> the entire dataset will be loaded on device at once.
    If train_ratio = 1 | 0 | None -> don't split for training and return just one dataloader
    """
    if not test_batch_size:
        test_batch_size = batch_size
        
    split = utils.split_data_and_metadata(
        data_file_path,
        metadata_file_path,
        train_ratio)
    
    from mmvae.data.datasets.CellCensusDataSet import CellCensusDataset, collate_fn

    if device:
        split[0][0] = split[0][0].to(device)
        if len(split) == 2:
            split[1][0] = split[1][0].to(device)
    
    train_dataset = CellCensusDataset(*split[0])
    from torch.utils.data import DataLoader
    if len(split) == 1:
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    test_dataset = CellCensusDataset(*split[1])
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        ),
        DataLoader(
            test_dataset,
            shuffle=shuffle,
            batch_size=test_batch_size,
            collate_fn=collate_fn,
        )
    )