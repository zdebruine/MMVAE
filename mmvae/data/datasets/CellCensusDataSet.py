from torch.utils.data import Dataset
import torch
import pandas as pd

def collate_fn(batch):
    tensors, metadata = zip(*batch)
    return torch.vstack(tensors), metadata
    
class CellCensusDataset(Dataset):
    """CellCensusDataset from a single .npz"""
    
    def __init__(self, data: torch.Tensor, labels: pd.DataFrame):
        super(CellCensusDataset, self).__init__()
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.labels is not None:
            labels = self.labels.iloc[idx, :]
        else:
            labels = None
        return data, labels