import torch
import torch.nn.functional as F
from torch.nn.modules import Module
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MappedCellCensusDataLoader
import numpy as np

class HumanVAETrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: HumanVAE.Model
    dataloader: MappedCellCensusDataLoader

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size # Defined before super().__init__ as configure_* is called on __init__
        super(HumanVAETrainer, self).__init__(*args, **kwargs)

    def configure_model(self) -> Module:
        return HumanVAE.configure_model(self.device, self.writer, init_weights=True) 
    
    def configure_dataloader(self):
        return MappedCellCensusDataLoader(
            batch_size=self.batch_size,
            device=self.device,
            file_path='/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz',
            load_all=True
        )
    
    def configure_optimizers(self):
        l2_reg = 1e-5
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=1e-5, weight_decay=l2_reg),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=1e-5, weight_decay=l2_reg),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=1e-5,  weight_decay=l2_reg)
        }
    
    def configure_schedulers(self):
        from torch.optim.lr_scheduler import StepLR
        return {}
        return {
            'encoder': StepLR(self.optimizers['encoder'], step_size=30, gamma=0.1),
            'decoder': StepLR(self.optimizers['decoder'], step_size=30, gamma=0.1),
            'shr_vae': StepLR(self.optimizers['shr_vae'], step_size=30, gamma=0.1)
        }

    def train(self, epochs, load_snapshot=False):
        self.total_epochs = epochs
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)
    
    def train_epoch(self, epoch):
        for train_data in self.dataloader:
            self.batch_iteration += 1
            self.train_trace_complete(train_data, epoch)

    def train_trace_complete(self, train_data: torch.Tensor, epoch):

        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()
        
        x_hat, mu, logvar = self.model(train_data)
        dense_train_data = train_data.to_dense()
        r2_score = utils.calculate_r2(dense_train_data, x_hat.detach())
        recon_loss = F.l1_loss(x_hat, dense_train_data)
        kl_loss = utils.kl_divergence(mu, logvar)
        kl_weight = utils.cyclic_annealing(self.batch_iteration, 2 * len(self.dataloader))
        unweighted_loss = recon_loss + kl_loss
        loss = recon_loss + (kl_loss * kl_weight)
        loss.backward()
    
        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()
        
        self.writer.add_scalar('Metric/R2_Score', r2_score, self.batch_iteration)
        self.writer.add_scalar('Metric/KL_Weight', kl_weight, self.batch_iteration)
        self.writer.add_scalar('Loss/KL', kl_loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/WeightedTotalLoss', loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/UnWeightedTotalLoss', unweighted_loss.item(), self.batch_iteration)
        self.writer.flush()