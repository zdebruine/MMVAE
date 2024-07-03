from collections import OrderedDict
import torch
import torch.nn as nn
from ._vae import VAE
import pandas as pd

from ..data import dropfilters

from .mixins import VAEMixIn, HeWeightInitMixIn
        
    
class DFVAE(VAEMixIn, HeWeightInitMixIn, nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        ld = self.latent_dim
        
        dataset_dfs = None
        with open(dropfilters.UNIQUE_DATASETS_PATH, 'r') as file:
            df = pd.read_csv(file, header=None)
            dataset_dfs = {row[0].replace('.', '_'): Block(ld) for row in df.itertuples(index=False)}
            
        assay_dfs = None
        with open(dropfilters.UNIQUE_ASSAYS_PATH, 'r') as file:
            df = pd.read_csv(file, header=None)
            assay_dfs = {row[0].replace('.', '_'): Block(ld) for row in df.itertuples(index=False)}
        
        donor_dfs = None
        with open(dropfilters.UNIQUE_DONORS_PATH, 'r') as file:
            df = pd.read_csv(file, header=None)
            donor_dfs = {row[0].replace('.', '_'): Block(ld) for row in df.itertuples(index=False)}

        self.dataset_dfs = nn.ModuleDict(dataset_dfs)
        self.assay_dfs = nn.ModuleDict(assay_dfs)
        self.donor_dfs = nn.ModuleDict(donor_dfs)
        
        self.init_weights()
        
    def after_reparameterize(self, z, metadata):
        
        if isinstance(metadata, pd.DataFrame):
            pass
        elif metadata == None:
            import warnings
            warnings.warn("After reparameterize has no affect as metadata cannot be found dfblocks are no-op")
            return z
            
        dataset_masks = self.generate_masks('dataset_id', metadata)
        assay_masks = self.generate_masks('assay', metadata)
        donor_id_masks = self.generate_masks('donor_id', metadata)
        
        z = self.forward_masks(z, self.dataset_dfs, dataset_masks)
        z = self.forward_masks(z, self.assay_dfs, assay_masks)
        z = self.forward_masks(z, self.donor_dfs, donor_id_masks)
        
        return z
    
    def generate_masks(self, filter_key: str, metadata: pd.DataFrame):
        values = metadata[filter_key]
        masks = {}
        
        for val in values:
            if val in masks:
                continue
            mask = (val == values[:])
            masks[val] = mask.to_list()
            
        return masks
    
    def forward_masks(self, z, dfs, masks):
        df_sub_batches = []
        for key, mask in masks.items():
            key = key.replace('.', '_')
            if not key in dfs:
                print(f"{key} not in {dfs}")
                raise RuntimeError()
            module = dfs[key]
            x = module(z[mask])
            df_sub_batches.append(x)
        return torch.cat(df_sub_batches, dim=0)
    
    def configure_optimizers(self):
        return torch.optim.Adam([
            { 'params': self.encoder.parameters(), 'lr': self.encoder_lr},
            { 'params': self.decoder.parameters(), 'lr': self.decoder_lr},
            { 'params': self.fc_mean.parameters(), 'lr': self.fc_mean_lr},
            { 'params': self.fc_var.parameters(), 'lr': self.fc_var_lr},
            { 'params': self.dataset_dfs.parameters(), 'lr': self.fc_mean_lr},
            { 'params': self.donor_dfs.parameters(), 'lr': self.fc_mean_lr},
            { 'params': self.assay_dfs.parameters(), 'lr': self.fc_mean_lr},
        ])