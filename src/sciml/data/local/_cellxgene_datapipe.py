import random
from typing import Callable, Iterable, Literal, Optional, Union
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle

from sciml.utils.constants import REGISTRY_KEYS as RK

import torch
from torchdata.datapipes.iter import FileLister, IterDataPipe, Zipper, Multiplexer
from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import sharding


@functional_datapipe("load_matrix_and_dataframe")
class LoadIndexMatchedCSRMatrixAndDataFrameDataPipe(IterDataPipe):
    
    def __init__(self, source_datapipe, verbose: bool = False):
        super().__init__()
        self.source_dp = source_datapipe
        self.verbose = verbose
        
    def __iter__(self):
        """Split incoming tuple from FileLister and load scipy .npz"""
        for npz_path, metadata_path in self.source_dp:
            
            if self.verbose:
                print(f"Loading file path: {npz_path}, {metadata_path}", flush=True)
            
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
            
@functional_datapipe("shuffle_matrix_and_dataframe")
class ShuffleCSRMatrixAndDataFrameDataPipe(IterDataPipe):
    
    def __init__(self, source_datapipe):
        super().__init__()
        self.source_dp = source_datapipe
        
    def __iter__(self):
        for sparse_matrix, dataframe in self.source_dp:
            
            permutation = np.random.permutation(sparse_matrix.shape[0])
            
            dataframe = dataframe.iloc[permutation].reset_index(drop=True)
            sparse_matrix = sparse_matrix[permutation]
            
            yield (sparse_matrix, dataframe)

@functional_datapipe("batch_csr_matrix_and_dataframe") 
class SparseCSRMatrixBatcherDataPipe(IterDataPipe):
    """
    Yields batches of torch.sparse_csr_tensor's of batch_size from sparse matrice from input datapipe.
    Args: 
     - batch_size: Size of the row 
     - drop_last: Drops last batch to ensure batches of equal size
     - tensor_func: function to create tensor ie. (torch.tensor)
    """
    def __init__(
        self, 
        source_datapipe,
        batch_size: int, 
        allow_partials: bool = False,
        return_dense: bool = False
    ):
        super(SparseCSRMatrixBatcherDataPipe, self).__init__()
        
        self.source_datapipe = source_datapipe
        self.batch_size = batch_size
        self.allow_partials = allow_partials
        self.return_dense = return_dense

    def __iter__(self):
        for sparse_matrix, dataframe in self.source_datapipe:
            
            n_samples = sparse_matrix.shape[0]
            
            for i in range(0, n_samples, self.batch_size):
                data_batch = sparse_matrix[i:i + self.batch_size]
                
                if self.allow_partials and not data_batch.shape[0] == self.batch_size:
                    continue
                
                tensor = torch.sparse_csr_tensor(
                    crow_indices=data_batch.indptr, 
                    col_indices=data_batch.indices, 
                    values=data_batch.data, 
                    size=data_batch.shape)
                
                if isinstance(dataframe, pd.DataFrame):
                    metadata = dataframe.iloc[i:i + self.batch_size]

                if self.return_dense:
                    tensor = tensor.to_dense()
                    
                yield tensor, metadata
                
@functional_datapipe("transform")
class TransformDataPipe(IterDataPipe):
    
    def __init__(self, source_datapipe, transform_fn: Callable):
        self.source_datapipe = source_datapipe
        self.transform_fn = transform_fn
        
    def __iter__(self):
        for source in self.source_datapipe:
            yield self.transform_fn(source)
            

class SpeciesDataPipe(IterDataPipe):
    
    def __init__(
        self, 
        directory_path: str, 
        npz_masks: Union[str, list[str]], 
        metadata_masks: Union[str, list[str]],
        batch_size: int,
        shuffle: bool = True,
        return_dense: bool = False,
        verbose: bool = False,
        transform_fn: Optional[Callable] = None,
    ):
        super(SpeciesDataPipe, self).__init__()
        
        # Create file lister datapipe for all npz files in dataset
        npz_paths_dp = FileLister(
            root=directory_path, 
            masks=npz_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False
        )

        # Create file lister datapipe for all metadata files 
        metadata_paths_dp = FileLister(
            root=directory_path, 
            masks=metadata_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False
        )
        
        self.zipped_paths_dp = Zipper(npz_paths_dp, metadata_paths_dp)

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
                
        self.batch_size = batch_size
        self.return_dense = return_dense
        self.verbose = verbose
        # Prevent overriding IterDataPipe shuffle
        self._shuffle = shuffle
        self.transform_fn = transform_fn

    # def _set_seed(self, seed: Optional[int] = None):
    #     if seed is None:
    #         seed = random.randint(0, 1000)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)
                
    def __iter__(self):
        # self.set_seed()
        
        dp = self.zipped_paths_dp.sharding_filter()
        
        if self._shuffle:
            dp = dp.shuffle()
        
        dp = dp.load_matrix_and_dataframe(self.verbose)
        
        if self._shuffle:
            dp = dp.shuffle_matrix_and_dataframe()
        
        dp = dp.batch_csr_matrix_and_dataframe(self.batch_size, return_dense=self.return_dense)
        
        # thought process on removal
        # the matrix is already completly shufled before batching
        # the shuffle dp holds a buffer and shuffles the buffer by pulling in samples to shuffle
        # this could cause it to hold more npz in memory then desired
        
        # if self._shuffle:
        #     dp = dp.shuffle()
        
        if callable(self.transform_fn):
            dp = dp.transform(self.transform_fn)
        
        try:
            yield from dp
        except Exception as e:
            print(f"Error during iteration: {e}")
            raise
        finally:
            # Ensure all resources are properly cleaned up
            pass

class RandomSelectDataPipe(IterDataPipe):
    def __init__(self, *datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        while iterators:
            selected_iterator = random.choice(iterators)
            try:
                yield next(selected_iterator)
            except StopIteration:
                # Remove exhausted iterator
                iterators.remove(selected_iterator)
        raise StopIteration
        
        
class MultiSpeciesDataPipe(IterDataPipe):
    
    def __init__(self, *species: SpeciesDataPipe, selection_fn: Union[Literal["random"], Literal["sequential"]] = 'random'):
        
        if selection_fn == "sequential":
            self.datapipe = Multiplexer(*species)
        elif selection_fn == "random":
            self.datapipe = RandomSelectDataPipe(*species)
        
    def __iter__(self):
        yield from self.datapipe
        
