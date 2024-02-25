import torch
import torch.nn.functional as F
from torch.nn.modules import Module
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE_gan as HumanVAE
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MappedCellCensusDataLoader
import numpy as np

lr = 0.0001
discrim_rato = 25
REAL = 1
FAKE = 0
KL_ANNEALING_STEPS = 50
DISCRIM_ANNEALING_STEPS = KL_ANNEALING_STEPS * discrim_rato
TPR = []
FPR = []

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
        self.annealing_steps = KL_ANNEALING_STEPS

    def configure_model(self) -> Module:
        return HumanVAE.configure_model() 
    
    def configure_dataloader(self):
        return MappedCellCensusDataLoader(
            batch_size=self.batch_size,
            device=self.device,
            file_path='/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz',
            load_all=True
        )
    
    def configure_optimizers(self):
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=lr),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=lr),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=lr),
            'discrim': torch.optim.Adam(self.model.expert.discriminator.parameters(), lr=lr/discrim_rato)
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
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)
        np.save('TPR.npy', TPR)
        np.save('FPR.npy', FPR)
        print("TPR and FPR saved")
    
    def train_epoch(self, epoch):
        for train_data in self.dataloader:
            self.batch_iteration += 1
            self.train_trace_complete(train_data, epoch)

    def train_trace_complete(self, train_data: torch.Tensor, epoch):
        # Zero All Gradients
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()
        self.optimizers['discrim'].zero_grad()

        ## Train Discriminator

        # train discriminator with real data **use l1 & 2
        real_pred = self.model.expert.discriminator(train_data)
        real_loss = F.l1_loss(real_pred, torch.ones_like(real_pred))
        real_loss.backward()

        # gen fake data
        with torch.no_grad():
            fake_data, _, _ = self.model(train_data)

        # train discriminator with fake data
        fake_pred = self.model.expert.discriminator(fake_data)
        fake_loss = F.l1_loss(fake_pred, torch.zeros_like(fake_pred))
        fake_loss.backward()

        self.optimizers['discrim'].step()

        # Forwad Pass Over Entire Model
        x_hat, mu, var = self.model(train_data)

        
        discrim_pred = self.model.expert.discriminator(x_hat)
        gen_loss = F.l1_loss(discrim_pred, torch.ones_like(discrim_pred))
        gan_weight = min(1.0, epoch / DISCRIM_ANNEALING_STEPS)

        recon_loss = F.l1_loss(x_hat, train_data.to_dense())
        # Shared VAE Loss
        kl_loss = utils.kl_divergence(mu, var)
        kl_weight = min(1.0, epoch / self.annealing_steps)
        # add generator weight
        loss = (gen_loss * gan_weight) + recon_loss + (kl_loss * kl_weight)
        loss.backward()

        tpr, fpr = utils.get_batch_tpr_fpr(self.model.expert.discriminator, train_data, x_hat)

        TPR.append(tpr)
        FPR.append(fpr)
    
        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()

        self.writer.add_scalar('Loss/KL', kl_loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/ReconstructionFromTrainingData', loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/RealDataLoss', real_loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/FakeDataLoss', fake_loss.item(), self.batch_iteration)
        self.writer.add_scalar('Loss/GeneratorLoss', gen_loss.item(), self.batch_iteration)
        self.writer.add_scalar('TPR', tpr, self.batch_iteration)
        self.writer.add_scalar('FPR', fpr, self.batch_iteration)

