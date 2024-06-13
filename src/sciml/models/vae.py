from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from . import utils

from sciml.utils.constants import REGISTRY_KEYS as RK


 
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
        self.vae = vae
        
        # Register kl_weight as buffer
        self.register_buffer('kl_weight', torch.tensor(self.hparams.kl_weight, requires_grad=False))
        
        # container for z embeddings/metadata/global_step
        self.z_val = []
        self.z_val_metadata = []
        self.validation_epoch_end = -1
    
    @property
    def plot_z_embeddings(self):
        return self.hparams.plot_z_embeddings and not self.trainer.sanity_checking
        
    def forward(self, x):
        return self.vae(x)
            
    def criterion(self, x, forward_outputs):
        # Mean Square Error of input batch to reconstructed batch
        recon_loss = F.mse_loss(forward_outputs[RK.X_HAT], x, reduction='mean')
        # Kl Divergence from posterior distribution to normal distribution
        kl_loss = utils.kl_divergence(forward_outputs[RK.QZM], forward_outputs[RK.QZV])
        loss = recon_loss + self.kl_weight * kl_loss
        return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss }
    
    def loss(self, batch_dict, return_outputs = False):
        
        # Call forward method on self
        forward_outputs = self(batch_dict)
        # Compute loss outputs based on criterion
        loss_outputs = self.criterion(batch_dict[RK.X], forward_outputs)
        # If forward outputs are needed return forward_outputs
        if return_outputs:
            return loss_outputs, forward_outputs
        return loss_outputs
    
    def training_step(self, batch_dict, batch_idx):

        # Compute loss outputes
        loss_outputs = self.loss(batch_dict)
        # Tag loss output keys with 'train_'
        loss_outputs = utils.tag_loss_outputs(loss_outputs, 'train')
        # Log loss outputs for every step and epoch to logger
        self.log_dict(loss_outputs, on_step=True, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
        # Return the train_loss for backwards pass
        return loss_outputs['train_loss']

    def validation_step(self, batch_dict, batch_idx):
        # Compute loss_outputs returning the forward_outputes if needed for plotting z embeddings
        if self.plot_z_embeddings:
            loss_outputs, forward_outputs = self.loss(batch_dict, return_outputs=self.plot_z_embeddings)
        else:
            loss_outputs = self.loss(batch_dict, return_outputs=False)
        # Tag loss output keys with 'val_'
        loss_outputs = utils.tag_loss_outputs(loss_outputs, 'val')
        
        # Prevent logging sanity_checking steps to logger
        # Not needed for validation and also throws error if batch_size not configured correctly
        if not self.trainer.sanity_checking:
            self.log_dict(loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

        # If plotting z embeddings
        if self.plot_z_embeddings:
            # Accumlate the Z tensors and associated metadata
            self.z_val.append(forward_outputs[RK.Z].detach().cpu())
            self.z_val_metadata.append(batch_dict.get(RK.METADATA))
    
    def on_validation_epoch_end(self):
        
        if self.plot_z_embeddings:
            # Record for global step
            self.validation_epoch_end += 1
            # Concatenate Z tensors
            embeddings = torch.cat(self.z_val, dim=0)
            # Concatenate metadata
            metadata = np.concatenate(self.z_val_metadata, axis=0)
            # get tensorboard SummaryWriter instance from trainer.logger
            writer = self.trainer.logger.experiment
            # Record z embeddings and metadata
            writer.add_embedding(
                mat=embeddings, 
                metadata=metadata.tolist(), 
                global_step=self.validation_epoch_end, 
                metadata_header=list(self.trainer.datamodule.hparams.obs_column_names))
            # Empty the Z tensors and metadata containers
            self.z_val = []
            self.z_val_metadata = []
            
    def test_step(self, batch, batch_idx):
        # Compute loss outputes
        loss_outputs = self.loss(batch)
        # Tag loss output keys with 'test_'
        loss_outputs = utils.tag_loss_outputs(loss_outputs, 'test')
        # Log loss outputes on epoch and on every step
        self.log_dict(loss_outputs, on_step=True, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

    def predict_step(self, batch_dict, batch_idx):
        # Run forward pass on model
        forward_outputs = self(batch_dict)
        # Return the tensors from the keys specified in hparams
        return { 
            key: value for key, value in forward_outputs 
            if key in self.hparams.predict_keys
        }
        
    def get_latent_representations(
        self,
        adata,
        batch_size
    ):
        from sciml.data.server import AnnDataDataset, collate_fn
        from torch.utils.data import DataLoader
        
        from lightning.pytorch.trainer import Trainer
        
        dataset = AnnDataDataset(adata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        zs = []
        self.eval()
        with torch.no_grad():
            for batch_dict in dataloader:
                for key in batch_dict.keys():
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].to('cuda')
                    
                predict_outputs = self.predict_step(batch_dict, None)
                zs.append(predict_outputs[RK.Z])
        
        return torch.cat(zs).numpy()