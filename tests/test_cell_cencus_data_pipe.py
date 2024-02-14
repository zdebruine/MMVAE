import pytest
import scipy.sparse as sp
import numpy as np
from d_mmvae.data.pipes.CellCensus import shuffle_sparse_matrix

def test_shuffle_sparse_matrix():
    matrix = sp.csr_matrix(np.array([[1, 2], [3, 4]]))
    shuffled_matrix = shuffle_sparse_matrix(matrix)
    assert shuffled_matrix.shape == matrix.shape