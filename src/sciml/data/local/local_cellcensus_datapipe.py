from typing import Union
import numpy as np
import scipy.sparse as sp
from torchdata.datapipes.iter import FileLister, IterDataPipe, Zipper
import pandas as pd
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
import pickle

class ChunkloaderDataPipe(IterDataPipe):
    
    def __init__(self, directory_path: str, npz_masks: Union[str, list[str]], metadata_masks: Union[str, list[str]], verbose: bool = False):
        super(ChunkloaderDataPipe, self).__init__()
        self.npz_files = FileLister(
            root=directory_path, 
            masks=npz_masks,
            recursive=False,
            abspath=True,
            non_deterministic=True
        )
        
        self.metadata_files = FileLister(
            root=directory_path, 
            masks=metadata_masks,
            recursive=False,
            abspath=True,
            non_deterministic=True
        )
        if (verbose):
            print("Files for dataloader:")
            for file in self.npz_files:
                print(file)
                
        self.verbose = verbose
        self.npz_files = self.npz_files.open_files(mode='b')
        self.metadata_files = self.metadata_files.open_files(mode='b')
    
    def load(self, file_tuples):
        """Split incoming tuple from FileLister and load scipy .npz"""
        (npz_path, npz_file), (metadata_path, metadata_file) = file_tuples
        if self.verbose:
            print(f"Loading file path: {npz_path}, {metadata_path}")
        sparse_matrix = sp.load_npz(npz_file)
        dataframe = pickle.load(metadata_file)
        
        return (sparse_matrix, dataframe)
    
    def shuffle(self, inputs: tuple[sp.csr_matrix, pd.DataFrame]):
        sparse_matrix, dataframe = inputs
        permutation = np.random.permutation(sparse_matrix.shape[0])
        
        dataframe = dataframe.iloc[permutation].reset_index(drop=True)
        sparse_matrix = sparse_matrix[permutation]
        
        return (sparse_matrix, dataframe)
                
    def __iter__(self):
        return iter(
            Zipper(self.npz_files, self.metadata_files)
            .shuffle()
            .map(self.load)
            .map(self.shuffle)
        )
            
    
def LocalCellCensusDataPipe(directory_path: str, npz_mask: Union[str, list[str]], metadata_mask: Union[str, list[str]], batch_size: int, verbose=False, return_dense=False) -> IterDataPipe: # type: ignore
    """
    Pipeline built to load Cell Census sparse csr chunks. 

    Important Note: The sharding_filter is applied aftering opening files 
        to ensure no duplication of chunks between worker processes.
    """
    pipe = (ChunkloaderDataPipe(directory_path, npz_mask, metadata_mask, verbose=verbose)
        .sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING) # Prevents chunks from being duplicated across workers
        .batch_sparse_csr_matrix(batch_size, return_dense=return_dense)
        .shuffle()
    )
    return pipe