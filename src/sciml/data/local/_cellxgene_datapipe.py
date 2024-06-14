from typing import Union
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle

from sciml.utils.constants import REGISTRY_KEYS as RK

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
        for npz_path, metadata_path in self.source_dp:
            if self.verbose:
                print(f"Loading file path: {npz_path}, {metadata_path}")
            
            sparse_matrix = None
            metadata = None
            
            try:
                with open(npz_path, 'rb') as npz_file:
                    sparse_matrix = sp.load_npz(npz_file)
                
                with open(metadata_path, 'rb') as metadata_file:
                    if '.pkl' in metadata_path:
                        metadata = pickle.load(metadata_file)
            except Exception as e:
                print(f"Error loading files: {e}")
                raise
            
            yield (sparse_matrix, metadata)
            
@functional_datapipe("shuffle_matrix_and_metadata")
class ShuffleCSRMatrixAndMetadataDataPipe(IterDataPipe):
    
    def __init__(self, source_datapipe):
        super().__init__()
        self.source_dp = source_datapipe
        
    def __iter__(self):
        for sparse_matrix, dataframe in self.source_dp:
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
                yield {RK.X: tensor, RK.METADATA: metadata} 
                
            
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
        
        self.zipped_paths_dp = Zipper(self.npz_paths_dp, self.metadata_paths_dp).sharding_filter()
        
        # Sanity check that the metadata files and npz files are correlated
        # and all files are masked correctly
        chunk_paths = []
        for npz_path, metadata_path in self.zipped_paths_dp:
            chunk_paths.append(f"\n\t{npz_path}\n\t{metadata_path}")
        if not chunk_paths:
            raise RuntimeError("No files found for masks from file lister")
        
        if verbose:
            for path in chunk_paths:
                print(path)
                
        self.verbose = verbose
                
    def __iter__(self):
        try:
            yield from self.zipped_paths_dp \
                .shuffle() \
                .load_matrix_and_metadata(self.verbose) \
                .shuffle_matrix_and_metadata()
        except Exception as e:
            print(f"Error during iteration: {e}")
            raise
        finally:
            # Ensure all resources are properly cleaned up
            pass
        
class CellxgeneDataPipe(IterDataPipe):
    
    def __init__(
        self,
        directory_path: str, 
        npz_mask: Union[str, list[str]], 
        metadata_mask: Union[str, list[str]], 
        batch_size: int,
        return_dense=False,
        seed: int = 42,
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
            .batch_sparse_csr_matrix(batch_size, return_dense=return_dense) \
            .shuffle()
        self.seed = seed
        
    def __iter__(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield from self.datapipe