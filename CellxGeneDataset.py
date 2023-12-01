import random
import torch
from torch.utils.data import IterableDataset
from scipy import sparse as sp

"""
This file holds the classes to load CellxGene data into PyTorch sparse tensors
"""

class CellxGeneDataset(IterableDataset):
    """
    Inherits from IterableDataset and takes the batch size as a parameter.
    The object will load data chunks one at a time and return a sparse tensor
    based on the batch size on every __iter__ call.
    """
    def __init__(self, batch_size):
        # batch_size is passed to the dataset as no DataLoader is used atm
        super(CellxGeneDataset).__init__()
        # Store a list of filepaths to iter over and load data as needed
        self.chunk_paths = [f'/active/debruinz_project/CellCensus/Python/chunk{n}_counts.npz' for n in range(1, 100)]
        self.batch_size = batch_size
    
    def __iter__(self):
        # Shuffle the paths on every epoch
        random.shuffle(self.chunk_paths)
        for path in self.chunk_paths:
            # data files are expected to be CSC sparse matricies
            chunk = sp.load_npz(path)
            """
            Since the batch size will not evenly split the chunk size, we drop some data.
            This ensures each batch is the same size and should help the model generalize.
            dropout_size = the 'leftover' chunk that doesn't fill a whole batch
            dropout = a randomly chosen index to increment the slicing op and skip data
            """
            dropout_size = chunk.shape[1] % self.batch_size
            # ensures that the dropout index will be hit during iteration
            dropout = random.randint(0, chunk.shape[1] // self.batch_size) * self.batch_size
            i = 0
            # while loop is used as the range() function generates constant step sizes
            while i < chunk.shape[1]:
                if i == dropout:
                    i += dropout_size
                else:
                    sliced = chunk[:, i:i+self.batch_size]
                    # yield returns a single tensor and then the function resumes execution from here when called again
                    yield torch.sparse_csc_tensor(sliced.indptr, sliced.indices, sliced.data, sliced.shape)
                    i += self.batch_size