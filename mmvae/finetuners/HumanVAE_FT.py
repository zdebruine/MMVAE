import torch
import torch.nn.functional as F
from torch.nn.modules import Module
import mmvae.trainers.utils as utils
import mmvae.finetuners.FineTuneModel as HumanVAE_FT
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MappedCellCensusDataLoader

class HumanVAE_FineTune(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: HumanVAE_FT.Model
    dataloader: MappedCellCensusDataLoader

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size # Defined before super().__init__ as configure_* is called on __init__
        super(HumanVAE_FineTune, self).__init__(*args, **kwargs)
        self.model.to(self.device)
        self.annealing_steps = 50

    def configure_model(self) -> Module:
        return HumanVAE_FT.configure_model() 
    
    def configure_dataloader(self):
        return MappedCellCensusDataLoader(
            batch_size=self.batch_size,
            device=self.device,
            file_path='/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz',
            load_all=True
        )
    
    def configure_optimizers(self):
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=0.0001),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=0.0001),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=.00001)
        }
    
    def configure_schedulers(self):
        from torch.optim.lr_scheduler import StepLR
        return {}
    
    def train(self, epochs, load_snapshot=False):
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)
    
    def train_epoch(self, epoch):
        
        for train_data in self.dataloader:
            print(f"Training on batch: {self.batch_iteration}, train_data: {train_data.shape} ")
            
            self.batch_iteration += 1
            self.train_trace_complete(train_data, epoch)

    def train_trace_complete(self, train_data: torch.Tensor, epoch):
        # Zero All Gradients
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()

        # Forwad Pass Over Entire Model
        x_hat, mu, var = self.model(train_data)
        recon_loss = F.l1_loss(x_hat, train_data.to_dense())
        
        # Shared VAE Loss
        kl_loss = utils.kl_divergence(mu, var)
        kl_weight = min(1.0, epoch / self.annealing_steps)
        loss = recon_loss + (kl_loss * kl_weight)
        loss.backward()
    
        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()

        self.writer.add_scalar('Loss/KL', kl_loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/ReconstructionFromTrainingData', loss.item(), self.batch_iteration)