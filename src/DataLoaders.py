import torch
import numpy as np
from scipy.sparse import load_npz
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import torchdata.datapipes as dp
from scipy.sparse import csr_matrix

@dp.functional_datapipe("batch_sparse_matrix")
class SparseMatrixBatcherDataPipe(dp.iter.IterDataPipe):
    def __init__(self, source_datapipe, batch_size):
        self.source_datapipe = source_datapipe
        self.batch_size = batch_size

    def __iter__(self):
        for sparse_matrix in self.source_datapipe:
            num_rows = sparse_matrix.shape[0]
            for i in range(0, num_rows, self.batch_size):
                batch = sparse_matrix[i:i + self.batch_size]
                yield torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape)

class CellCensusPipeLine:

    @staticmethod
    def load_sparse_matrix(file_tuple):
        path, file = file_tuple
        return load_npz(file)
    
    @staticmethod
    def shuffle_sparse_matrix(sparse_matrix: csr_matrix):
        shuffled_indices = np.random.permutation(sparse_matrix.shape[0])
        # Reorder the rows of the matrix according to the shuffled indices
        return sparse_matrix[shuffled_indices]
        
    @staticmethod
    def batch_and_convert_to_torch_sparse(sparse_matrix, batch_size):
        # Logic to break the sparse matrix into smaller batches
        # and convert each batch to torch.sparse_csr_tensor
        num_rows = sparse_matrix.shape[0]
        for i in range(0, num_rows, batch_size):
            batch = sparse_matrix[i:i + batch_size]
            yield torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape)

    @staticmethod
    def setup_pipeline(directory_path: str, masks: list[str], batch_size: int) -> dp.iter.IterDataPipe:
        return (
            dp.iter.FileLister(root=directory_path, masks=masks, recursive=False)
            .shuffle()
            .open_files(mode='rb')
            .map(CellCensusPipeLine.load_sparse_matrix)
            .map(CellCensusPipeLine.shuffle_sparse_matrix)
            .batch_sparse_matrix(batch_size)
        )

class CellCensusDataLoader(DataLoader2):

    def __init__(self, batch_size: int, directory_path: str, masks: list[str], num_workers: int):
        super(CellCensusDataLoader, self).__init__(
            datapipe=CellCensusPipeLine.setup_pipeline(directory_path, masks, batch_size),
            reading_service=MultiProcessingReadingService(num_workers=num_workers),
        )


batch_size = 32
directory_path = '/active/debruinz_project/tony_boos/csr_chunks/'
loader = CellCensusDataLoader(batch_size, directory_path, ['chunk*[1-3].npz'], 3)

for load in loader:
    pass