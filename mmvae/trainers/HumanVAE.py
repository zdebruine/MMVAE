import torch
import torch.nn.functional as F
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MappedCellCensusDataLoader
from datetime import datetime

class HumanVAETrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: HumanVAE.Model
    dataloader: MappedCellCensusDataLoader

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size # Defined before super().__init__ as configure_* is called on __init__
        super(HumanVAETrainer, self).__init__(*args, **kwargs)
        self.model.to(self.device)

    def configure_dataloader(self):
        return MappedCellCensusDataLoader(
            self.batch_size,
            self.device,
            '/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz'
        )

    def configure_model(self):
        return HumanVAE.configure_model()
    
    def configure_optimizers(self):
        weight_decay = 1e-5
        weight_decay = 0.0
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=0.0001, weight_decay=weight_decay),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=0.0001, weight_decay=weight_decay),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=.00001)
        }
    
    def configure_schedulers(self):
        from torch.optim.lr_scheduler import StepLR
        return {}
        return {
            'encoder': StepLR(self.optimizers['encoder'], step_size=30, gamma=0.1),
            'decoder': StepLR(self.optimizers['decoder'], step_size=30, gamma=0.1),
            'shr_vae': StepLR(self.optimizers['shr_vae'], step_size=30, gamma=0.1)
        }
    
    def expert_discriminator_loss(self, data, expert, real=True):
        output = expert.discriminator(data)
        labels = torch.ones(output.size(), device=self.device) if real else torch.zeros(output.size(), device=self.device)
        loss = F.binary_cross_entropy(output, labels)
        return loss
    
    def check_zero_gradients(self):
        zero_grad_layers = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                zero_grad_layers.append((name, param.grad))
        return zero_grad_layers
    
    def train(self, epochs, load_snapshot=False):
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)
    
    def train_epoch(self, epoch):
        for train_data in self.dataloader:
            self.batch_iteration += 1
            #self.train_trace_expert_autoencoder(self.batch_iteration, train_data)
            self.train_trace_complete(train_data)

    def train_trace_expert_autoencoder(self, train_data: torch.Tensor):
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()
    
        x = self.model.expert(train_data)
        
        loss = F.l1_loss(x, train_data.to_dense())
        loss.backward()

        self.writer.add_scalar('Expert_Loss/Reconstruction Loss (l1_loss)', loss.item(), self.batch_iteration)

        for model in ('encoder', 'decoder'):
            self.optimizers[model].step()
            if model in self.schedulers:
                self.schedulers[model].step()

    def train_trace_complete(self, train_data: torch.Tensor):
        # Zero All Gradients
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()

        # Forwad Pass Over Entire Model
        x_hat, mu, var = self.model(train_data)
        recon_loss = F.l1_loss(x_hat, train_data.to_dense())
        # Shared VAE Loss
        kl_loss = utils.kl_divergence(mu, var)
        kl_weight = 0
        loss = recon_loss + (kl_loss * kl_weight)
        loss.backward()
    
        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()

        #self.writer.add_scalar('Loss/Expert_Recon', expert_recon_loss.item(), self.batch_iteration)
        #self.writer.add_scalar('Loss/VAE', vae_recon.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/KL', kl_loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/ReconstructionFromTrainingData', loss.item(), self.batch_iteration)