import numpy as np
import scipy.sparse as sp
from torchdata.datapipes.iter import FileLister, IterDataPipe

                
def load_sparse_matrix(file_tuples):
    """Split incoming tuple from FileLister and load scipy .npz"""
    path, file = file_tuples
    return sp.load_npz(file)

def shuffle_sparse_matrix(sparse_matrix: sp.csr_matrix):
    shuffled_indices = np.random.permutation(sparse_matrix.shape[0])
    return sparse_matrix[shuffled_indices]
    
def CellCensusPipeLine(*args, directory_path: str = None, masks: list[str] = None, batch_size: int = None) -> IterDataPipe: # type: ignore
    """
    Pipeline built to load Cell Census sparse csr chunks. 

    Important Note: The sharding_filter is applied aftering opening files 
        to ensure no duplication of chunks between worker processes.
    """
    return (FileLister(
        root=directory_path, 
        masks=masks,
        recursive=False,
        non_deterministic=True)
        .shuffle()
        .open_files(mode='rb')
        .sharding_filter() # Prevents chunks from being duplicated across workers
        .map(load_sparse_matrix)
        .map(shuffle_sparse_matrix)
        .batch_sparse_csr_matrix(batch_size)
        .attach_to_output(*args)
    )
