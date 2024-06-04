import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from . import utils
import time
from lightning.pytorch.loggers import TensorBoardLogger

from sciml._constant import REGISTRY_KEYS as RK
        

class VAE(pl.LightningModule):

    def __init__(
        self, 
        encoder_layers = [60664, 1024, 512], 
        latent_dim=256, 
        decoder_layers = [256, 512, 1024, 60664], 
        learning_rate = 1e-4, 
        predict_keys = [RK.X_HAT, RK.Z],
        kl_weight=1.0
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters(ignore=[])

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        self.mean = self.build_mean()
        self.var = self.build_var()
      
        self.register_buffer('kl_weight', torch.tensor(self.hparams.kl_weight))
        
    def _build_encoder(self):
        layers = []
        n_in = self.hparams.encoder_layers[0]
        for n_out in self.hparams.encoder_layers[1:]:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    build_encoder = _build_encoder
    
    def _build_mean_var(self):
        return nn.Linear(self.hparams.encoder_layers[-1], self.hparams.latent_dim)
    
    build_mean = _build_mean_var
    build_var = _build_mean_var
    
    def _build_decoder(self):
        layers = []
        n_in = self.hparams.latent_dim
        for n_out in self.hparams.decoder_layers:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
        
    build_decoder = _build_decoder

    def configure_optimizers(self):
        super().configure_optimizers()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def encode(self, x):
        return self.encoder(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps * std
    
    def decode(self, x):
        return self.decoder(x)
    
    def before_reparameterize(self, x):
        return x
    
    def after_reparameterize(self, x, metadata):
        return x
    
    def forward(self, forward_inputs):
        x = forward_inputs.get(RK.X)
        y = forward_inputs.get(RK.Y)
        metadata = forward_inputs.get(RK.METADATA)
        
        x = self.encode(x)
        x = self.before_reparameterize(x)
        mu = self.mean(x)
        log_var = self.var(x)
        z = self.reparameterize(mu, log_var)
        x = self.after_reparameterize(z, metadata)
        x_hat = self.decode(x)
        
        return {
            RK.X_HAT: x_hat, 
            RK.Z: z, 
            RK.QZM: mu, 
            RK.QZV: log_var
        }
        
    def criterion(self, x, forward_outputs):
        recon_loss = F.mse_loss(forward_outputs[RK.X_HAT], x, reduction='mean')
        kl_loss = utils.kl_divergence(forward_outputs[RK.QZM], forward_outputs[RK.QZV])
        loss = recon_loss + self.kl_weight * kl_loss
        return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss }
    
    def loss(self, forward_inputs):
        forward_outputs = self(forward_inputs)
        loss_outputs = self.criterion(forward_inputs[RK.X], forward_outputs)
        return loss_outputs
    
    def training_step(self, batch, batch_idx):

        loss_outputs = self.loss(batch)
        
        self.log_dict({
            f"train_{key}": value
            for key, value in loss_outputs.items()
        }, on_step=True, on_epoch=True, logger=True)

        return loss_outputs

    def validation_step(self, batch, batch_idx):
        
        loss_outputs = self.loss(batch)

        return {
            f"val_{key}": value
            for key, value in loss_outputs.items()
        }
    
    def predict_step(self, batch, batch_idx):
        forward_outputs = self(batch)
        return { 
            key: value for key, value in forward_outputs 
            if key in self.hparams.predict_keys
        }
            
    def get_latent_representations(
        self,
        adata,
        batch_size
    ):
        from sciml.data import AnnDataDataset
        from torch.utils.data import DataLoader
        
        from lightning.pytorch.trainer import Trainer
        
        dataset = AnnDataDataset(adata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.hparams.num_workers)

        zs = []
        self.eval()
        for tensors in dataloader:
            predict_outputs = self.predict_step(tensors, None)
            zs.append(predict_outputs[RK.Z])
        
        return torch.cat(zs).numpy()
    
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
        
    
class DFVAE(VAE):
    
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