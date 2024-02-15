import pytest
import torch
import scipy.sparse as sp
import numpy as np
from mmvae.data.pipes.CellCensus import shuffle_sparse_matrix

def test_shuffle_sparse_matrix():
    matrix = sp.csr_matrix(np.array([[1, 2], [3, 4]]))
    shuffled_matrix = shuffle_sparse_matrix(matrix)
    assert shuffled_matrix.shape == matrix.shape

# def test_batches_are_different():
#     data_loader = CellCensusDataLoader(
#         'human', 
#         directory_path="/active/debruinz_project/CellCensus_3M/",
#         masks=['*human_chunk_1*', '*human_chunk_2*', '*human_chunk_3*'], 
#         batch_size=32, 
#         num_workers=3
#     )
#     data = iter(data_loader)
#     (chunk, _) = next(data)
#     chunk = chunk.to_dense()
#     for (next_chunk, _) in data:
#         assert not torch.equal(next_chunk.to_dense(), chunk)
    
# def test_cell_randmoness():
#     data_loader = CellCensusDataLoader( 
#         directory_path="/active/debruinz_project/CellCensus_3M/",
#         masks=['*human_chunk_1*', '*human_chunk_2*', '*human_chunk_3*'], 
#         batch_size=32, 
#         num_workers=3
#     )
#     chunk1 = next(iter(data_loader))
#     chunk2 = next(iter(data_loader))
#     assert not torch.equal(chunk1.to_dense(), chunk2.to_dense())

# def test_metalabel_append():
#     data_loader = CellCensusDataLoader(
#         'human', 
#         directory_path="/active/debruinz_project/CellCensus_3M/",
#         masks=['*human_chunk_1*', '*human_chunk_2*', '*human_chunk_3*'], 
#         batch_size=32, 
#         num_workers=3
#     )
#     prev_batch = None
#     data, label = next(iter(data_loader))
#     assert label == 'human'
#     del data_loader