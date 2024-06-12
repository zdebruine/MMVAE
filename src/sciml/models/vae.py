from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from . import utils

from sciml.utils.constants import REGISTRY_KEYS as RK
from sciml.modules import BasicVAE


 
class VAEModel(pl.LightningModule):

    def __init__(
        self,
        vae: nn.Module,
        predict_keys = [RK.X_HAT, RK.Z],
        kl_weight=1.0,
        batch_size: int = 128,
        plot_z_embeddings: bool = False,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['vae'], logger=True)
        self.register_buffer('kl_weight', torch.tensor(self.hparams.kl_weight, requires_grad=False))
        self.vae = vae
        
        # container for z embeddings/metadata/global_step
        self.z_val = []
        self.z_val_metadata = []
        self.validation_epoch_end = -1
        
    def forward(self, x):
        return self.vae(x)
            
    def criterion(self, x, forward_outputs):
        # Mean Square Error of input batch to reconstructed batch with sum reduction
        recon_loss = F.mse_loss(forward_outputs[RK.X_HAT], x, reduction='sum')
        # Kl Divergence from posterior distribution to normal distribution
        kl_loss = utils.kl_divergence(forward_outputs[RK.QZM], forward_outputs[RK.QZV])
        loss = recon_loss + self.kl_weight * kl_loss
        return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss }
    
    def loss(self, batch_dict, return_outputs = False):
        
        forward_outputs = self(batch_dict)
        loss_outputs = self.criterion(batch_dict[RK.X], forward_outputs)
        
        if return_outputs:
            return loss_outputs, forward_outputs
        return loss_outputs
    
    def training_step(self, batch_dict, batch_idx):

        loss_outputs = self.loss(batch_dict)
        loss_outputs = utils.tag_loss_outputs(loss_outputs, 'train')

        self.log_dict(loss_outputs, on_step=True, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

        return loss_outputs['train_loss']

    def validation_step(self, batch_dict, batch_idx):

        loss_outputs, forward_outputs = self.loss(batch_dict, return_outputs=True)
        loss_outputs = utils.tag_loss_outputs(loss_outputs, 'val')
        
        if not self.trainer.sanity_checking:
            self.log_dict(loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
            
            if self.hparams.plot_z_embeddings:
                self.z_val.append(forward_outputs[RK.Z].detach().cpu())
                self.z_val_metadata.append(batch_dict.get(RK.METADATA))
    
    def on_validation_epoch_end(self):
        if self.hparams.plot_z_embeddings \
            and not self.trainer.sanity_checking \
            and len(self.z_val) > 0:
                
            self.validation_epoch_end += 1
            embeddings = torch.cat(self.z_val, dim=0)
            metadata = np.concatenate(self.z_val_metadata, axis=0)
            
            writer = self.trainer.logger.experiment
                
            writer.add_embedding(
                mat=embeddings, 
                metadata=metadata.tolist(), 
                global_step=self.validation_epoch_end, 
                metadata_header=list(self.trainer.datamodule.hparams.obs_column_names))
        
            self.z_val = []
            self.z_val_metadata = []
            
    def test_step(self, batch, batch_idx):
        
        loss_outputs = self.loss(batch)
        loss_outputs = utils.tag_loss_outputs(loss_outputs, 'test')
        
        self.log_dict(loss_outputs, on_step=True, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

    def predict_step(self, batch_dict, batch_idx):
        forward_outputs = self(batch_dict)
        return { 
            key: value for key, value in forward_outputs 
            if key in self.hparams.predict_keys
        }
        
    def get_latent_representations(
        self,
        adata,
        batch_size
    ):
        from sciml.data.server import AnnDataDataset
        from torch.utils.data import DataLoader
        
        from lightning.pytorch.trainer import Trainer
        
        dataset = AnnDataDataset(adata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=1)

        zs = []
        self.eval()
        with torch.no_grad():
            for tensors in dataloader:
                for tk in tensors.keys():
                    if isinstance(tensors[tk], torch.Tensor):
                        tensors[tk] = tensors[tk].to('cuda')
                    
                predict_outputs = self.predict_step(tensors, None)
                zs.append(predict_outputs[RK.Z])
        
        return torch.cat(zs).numpy()