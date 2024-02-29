from torch.utils.data import Dataset
import scipy.sparse as sp
import torch

def collate_fn(batch):
    return torch.vstack(batch)

class CellCensusDataset(Dataset):
    """CellCensusDataset from a single .npz"""
    
    def __init__(self, device: torch.dtype, file_path: str, load_all: bool):
        super(CellCensusDataset, self).__init__()
        
        if load_all:
            import warnings
            warnings.warn(f"Warning! You are going to be loading the entire dataset into {device}!")
            
        self.load_all = load_all
        sparse_sp_data = sp.load_npz(file_path)
        assert isinstance(sparse_sp_data, sp.csr_matrix)
        self.data = torch.sparse_csr_tensor(sparse_sp_data.indptr, sparse_sp_data.indices, sparse_sp_data.data, sparse_sp_data.shape)
        if load_all:
            self.data = self.data.to(device)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if not self.load_all:
            return data.to(self.device)
        return data