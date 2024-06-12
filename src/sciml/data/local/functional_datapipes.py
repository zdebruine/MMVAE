import torch
from torchdata.datapipes import iter, functional_datapipe

@functional_datapipe("attach_to_output")
class AttachToOutput(iter.IterDataPipe):
    """
    Attaches *args, **kwargs to input to yield tuple of (input, (*args, **kwargs))
    """
    def __init__(self, source_datapipe, *args, **kwargs):
        super(AttachToOutput, self).__init__()
        self.source_datapipe = source_datapipe
        self.args, self.kwargs = args, kwargs

    def __iter__(self):
        for source in self.source_datapipe:
            if self.kwargs:
                yield source, *self.args, self.kwargs
            else:
                yield source, *self.args
                
@functional_datapipe("batch_sparse_csr_matrix") 
class SparseCSRMatrixBatcherDataPipe(iter.IterDataPipe):
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
                metadata = dataframe.iloc[i:i + self.batch_size]
                
                if self.return_dense:
                    tensor = tensor.to_dense()
                yield (tensor, metadata)