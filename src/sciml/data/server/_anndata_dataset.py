import torch
from torch.utils.data import Dataset, DataLoader
import anndata
import scipy.sparse as sp

from sciml.utils.constants import REGISTRY_KEYS as RK

class AnnDataDataset(Dataset):
    def __init__(self, adata):
        self.data = adata.X
        self.labels = adata.obs['labels'] if 'labels' in adata.obs else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        labels = None
        if self.labels is not None:
            labels = self.labels[idx]
        return [self.data[idx], labels]
    
def collate_fn(data):
    
    sp_mats = [row[0] for row in data]
    labels = [row[1] for row in data]
    
    sp_mats = sp.vstack(sp_mats)
    tensor = torch.sparse_csr_tensor(sp_mats.indptr, sp_mats.indices, sp_mats.data, sp_mats.shape)
    
    return {
        RK.X: tensor,
        RK.METADATA: labels
    }
