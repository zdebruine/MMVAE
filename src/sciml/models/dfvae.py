import torch
import torch.nn as nn
from .vae import VAEModel


class DFBlock(nn.Module):
    
    def __init__(self, latent_dim: int, linear: bool = True, batch_norm: bool = False, non_linear = False, repeat=1):
        super().__init__()
        layers = []
        for _ in range(repeat):
            if linear:
                layers.append(nn.Linear(latent_dim, latent_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(latent_dim))
            if non_linear:
                layers.append(nn.ReLU())
 
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
        
    
class DFVAEModel(VAEModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        ld = self.hparams.latent_dim
        self.dataset_dfs = nn.ModuleDict({
            'block1': DFBlock(ld),
            'block2': DFBlock(ld),
            'block3': DFBlock(ld),
            'block4': DFBlock(ld),
            'block5': DFBlock(ld),
            'block6': DFBlock(ld),
        })
        
        self.assay_dfs = nn.ModuleDict({
            'block1': DFBlock(ld),
            'block2': DFBlock(ld),
            'block3': DFBlock(ld),
            'block4': DFBlock(ld),
            'block5': DFBlock(ld),
            'block6': DFBlock(ld),
        })
        
        self.donor_dfs = nn.ModuleDict({
            'block1': DFBlock(ld),
            'block2': DFBlock(ld),
            'block3': DFBlock(ld),
            'block4': DFBlock(ld),
            'block5': DFBlock(ld),
            'block6': DFBlock(ld),
        })
        
    def after_reparameterize(self, z, metadata):
        
        if metadata == None:
            raise RuntimeWarning("No metadata found after_reparameterize")
            
        dataset_masks = _generate_masks(1, metadata)
        assay_masks = _generate_masks(2, metadata)
        donor_id_masks = _generate_masks(3, metadata)
            
        z = _forward_masks(z, self.dataset_dfs, dataset_masks)
        z = _forward_masks(z, self.assay_dfs, assay_masks)
        z = _forward_masks(z, self.donor_dfs, donor_id_masks)
        
        return z
    
def _generate_masks(metadata_idx, metadata):
    masks = []
    masked_idxs = []
    for idx in range(len(metadata)):
        if idx in masked_idxs:
            continue
        mask = metadata[:][metadata_idx] == metadata[idx][metadata_idx]
        masks.append(mask)
        masked_idxs.extend(mask)
        
def _forward_masks(z, dfs, masks):
    forward_outputs = []
    for mask in masks:
        x = dfs[mask[0]](z(mask))
        forward_outputs.append(x)
    return torch.cat(forward_outputs, dim=0)