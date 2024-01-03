import threading
import queue
import torch
from torch.utils.data import Dataset
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import time
import numpy as np

class CellxGeneDataset(Dataset):
    def __init__(self, batch_size, buffer_size=6):
        self.chunk_paths = [f'/active/debruinz_project/tony_boos/csr_chunks/chunk_{n}.npz' for n in range(1, 11)]
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)

        self.loading_thread = threading.Thread(target=self.load_chunks)
        self.loading_thread.daemon = True
        self.loading_thread.start()

        self.batch_idx = 0
        self.chunks_loaded = 0
        self.chunk_size = None
        self.current_chunk = None
    
    @property
    def total_batches(self) -> int:
        return self.batch_idx // self.batch_size

    def load_chunks(self) -> None:
        np.random.shuffle(self.chunk_paths)
        for path in self.chunk_paths:
            # NOTE: the buffer should never be full because of block=True on buffer.put which does't
            # add to the buffer until slot available
            if self.buffer.full():
                print("Buffer is full")
                break
            
            chunk = self.load_chunk(path)
            self.buffer.put(chunk, block=True) # waits for available slot
        self.buffer.put(None) # Indicates end of buffer input

    def load_chunk(self, path) -> csr_matrix:
        # Logic to load and shuffle the chunk
        chunk = sp.load_npz(path)
        shuffled_index = np.random.permutation(chunk.shape[0])
        return chunk[shuffled_index, :]

    def __getitem__(self, _):
        
        if (self.current_chunk == None):
            self.current_chunk = self.buffer.get(timeout=60) # Peek at the first chunk in the buffer
            
        self.chunk_size = self.current_chunk.shape[1]
        batch = self.current_chunk[self.batch_idx:self.batch_idx + self.batch_size, :]
        self.batch_idx += self.batch_size
        
        # Optionally, move to the next chunk if the current one is exhausted
        if self.is_chunk_exhausted():
            self.current_chunk = None
            self.chunks_loaded += 1 # Increment chunk index
        
        return torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape)
    
    def is_chunk_exhausted(self):
        return self.batch_idx % self.chunk_size + self.batch_size > self.chunk_size
    
    def __len__(self): 
        if (self.chunk_size == None):
            self.chunk_size = 285341
        return len(self.chunk_paths) * (self.chunk_size // self.batch_size)
