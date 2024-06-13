from typing import Union
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle

import torch
from torchdata.datapipes.iter import FileLister, IterDataPipe, Zipper
from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import sharding



@functional_datapipe("load_matrix_and_metadata")
class LoadCSRMatrixAndMetadataDataPipe(IterDataPipe):
    
    def __init__(self, source_datapipe, verbose):
        super().__init__()
        self.source_dp = source_datapipe
        self.verbose = verbose
        
    def __iter__(self):
        """Split incoming tuple from FileLister and load scipy .npz"""
        for file_tuples in self.source_dp:
            (npz_path, npz_file), (metadata_path, metadata_file) = file_tuples
            if self.verbose:
                print(f"Loading file path: {npz_path}, {metadata_path}")
            sparse_matrix = sp.load_npz(npz_file)
        
            if '.pkl' in metadata_path:
                labels = pickle.load(metadata_file)
            else:
                labels = None
            
            yield (sparse_matrix, labels)
            
@functional_datapipe("shuffle_matrix_and_metadata")
class ShuffleCSRMatrixAndMetadataDataPipe(IterDataPipe):
    
    def __init__(self, source_datapipe):
        super().__init__()
        self.source_dp = source_datapipe
        
    def __iter__(self):
        for inputs in self.source_dp:
            sparse_matrix, dataframe = inputs
            permutation = np.random.permutation(sparse_matrix.shape[0])
            
            if isinstance(dataframe, pd.DataFrame):
                dataframe = dataframe.iloc[permutation].reset_index(drop=True)
                
            sparse_matrix = sparse_matrix[permutation]
            
            yield (sparse_matrix, dataframe)

@functional_datapipe("batch_sparse_csr_matrix") 
class SparseCSRMatrixBatcherDataPipe(IterDataPipe):
    """
    Yields batches of torch.sparse_csr_tensor's of batch_size from sparse matrice from input datapipe.
    Args: 
     - batch_size: Size of the row 
     - drop_last: Drops last batch to ensure batches of equal size
     - tensor_func: function to create tensor ie. (torch.tensor)
    """
    def __init__(self, source_datapipe, batch_size, drop_last = True, return_dense=False):
        super(SparseCSRMatrixBatcherDataPipe, self).__init__()
        
        self.source_datapipe = source_datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.return_dense = return_dense

    def __iter__(self):
        for sparse_matrix, dataframe in self.source_datapipe:
            drop = 0 if self.drop_last else -self.batch_size
            for i in range(0, sparse_matrix.shape[0] - drop, self.batch_size):
                
                data_batch = sparse_matrix[i:i + self.batch_size]
                tensor = torch.sparse_csr_tensor(data_batch.indptr, data_batch.indices, data_batch.data, data_batch.shape)
                if isinstance(dataframe, pd.DataFrame):
                    metadata = dataframe.iloc[i:i + self.batch_size]
                else:
                    metadata = None
                
                if self.return_dense:
                    tensor = tensor.to_dense()
                yield (tensor, metadata)
                
            
class ChunkloaderDataPipe(IterDataPipe):
    
    def __init__(self, directory_path: str, npz_masks: Union[str, list[str]], metadata_masks: Union[str, list[str]], verbose: bool = False):
        super(ChunkloaderDataPipe, self).__init__()
        
        # Create file lister datapipe for all npz files in dataset
        self.npz_paths_dp = FileLister(
            root=directory_path, 
            masks=npz_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False
        )
        
        # Create file lister datapipe for all metadata files 
        self.metadata_paths_dp = FileLister(
            root=directory_path, 
            masks=metadata_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False
        )
        
        # Sanity check that the metadata files and npz files are correlated
        # and all files are masked correctly
        if (verbose):
            for i, (npz_path, metadata_path) in enumerate(Zipper(self.npz_paths_dp, self.metadata_paths_dp)):
                print(f"Chunk {i}:\n\t{npz_path}\n\t{metadata_path}")
                
        self.verbose = verbose
                
    def __iter__(self):
        npz_files_dp = self.npz_paths_dp.open_files(mode='rb')
        metadata_files_dp = self.metadata_paths_dp.open_files(mode='rb')
        return iter(
            Zipper(npz_files_dp, metadata_files_dp )
            .shuffle()
            .load_matrix_and_metadata(self.verbose)
            .shuffle_matrix_and_metadata()
        )
    
class CellxgeneDataPipe(IterDataPipe):
    
    def __init__(
        self,
        directory_path: str, 
        npz_mask: Union[str, list[str]], 
        metadata_mask: Union[str, list[str]], 
        batch_size: int,
        return_dense=False,
        verbose=False, 
    ) -> IterDataPipe: # type: ignore
        """
        Pipeline built to load Cell Census sparse csr chunks. 
        
        Args:
        - directory_path: str, 
        - npz_mask: Union[str, list[str]], 
        - metadata_mask: Union[str, list[str]], 
        - batch_size: int,
        - return_dense (bool): Converts torch.sparse_csr_tensor batch to dense
        - verbose (bool): print npz and metata file pairs, 

        Important Note: The sharding_filter is applied aftering opening files 
            to ensure no duplication of chunks between worker processes.
        """
        super().__init__()
        self.datapipe = ChunkloaderDataPipe(directory_path, npz_mask, metadata_mask, verbose=verbose) \
            .sharding_round_robin_dispatch(sharding.SHARDING_PRIORITIES.MULTIPROCESSING) \
            .batch_sparse_csr_matrix(batch_size, return_dense=return_dense) \
            .shuffle()
        
    def __iter__(self):
        yield from self.datapipe