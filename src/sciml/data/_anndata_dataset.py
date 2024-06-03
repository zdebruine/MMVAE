import torch
from torch.utils.data import Dataset, DataLoader
import anndata

from sciml._constant import REGISTRY_KEYS as RK

class AnnDataDataset(Dataset):
    def __init__(self, adata):
        self.data = adata.X
        print(adata.obs.keys())
        self.labels = adata.obs['labels'] if 'labels' in adata.obs else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
        return sample
