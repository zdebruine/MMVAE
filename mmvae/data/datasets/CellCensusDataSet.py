from torch.utils.data import Dataset
import scipy.sparse as sp
import torch

def collate_fn(batch):
    return torch.vstack(batch)

class CellCensusDataset(Dataset):

    _initialized = False
    
    def __init__(self, device: torch.dtype, file_path='/active/debruinz_project/CellCensus_3M_Full/3m_human_full.npz'):
        super(CellCensusDataset, self).__init__()
        sparse_sp_data = sp.load_npz(file_path)
        assert isinstance(sparse_sp_data, sp.csr_matrix)
        self.data = torch.sparse_csr_tensor(sparse_sp_data.indptr, sparse_sp_data.indices, sparse_sp_data.data, sparse_sp_data.shape).to(device) # type: ignore
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]