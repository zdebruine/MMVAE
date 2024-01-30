import torchdata.dataloader2 as dl
from typing import Generator, Any
import d_mmvae.data.pipes as p
import torch
import random

class MultiModalLoader:
    """
    Stochastic sampler from dataloaders
    
    Args:
     - exhaust_all: exhaust all dataloaders to completion (default=True)
    """

    def __init__(self, *modals: dl.DataLoader2, exhaust_all=True):
        self.exhaust_all = exhaust_all
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
                    raise e
                del loaders[loader_idx]
            
class CellCensusDataLoader(dl.DataLoader2):
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
    def __init__(self, *args, directory_path: str = None, masks: list[str] = None, batch_size: int = None, num_workers: int = None):
        super(CellCensusDataLoader, self).__init__(
            datapipe=p.CellCensusPipeLine(*args, directory_path=directory_path, masks=masks, batch_size=batch_size),
            datapipe_adapter_fn=None,
            reading_service=dl.MultiProcessingReadingService(num_workers=num_workers)
        )