import torch
import torch.nn.functional as F
from torch.nn.modules import Module
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE_gan as HumanVAE
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MappedCellCensusDataLoader
import numpy as np
import gan_testing.meta_discriminator as meta_disc
import discriminators.annealing
import matplotlib.pyplot as plt
from mmvae.trainers.utils import get_roc_stats
import io
import PIL.Image
from torchvision.transforms import ToTensor

lr = 0.0001
discrim_rato = 5000

class HumanVAETrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: HumanVAE.Model
    dataloader: MappedCellCensusDataLoader

    def __init__(self, batch_size, lr, annealing_steps, discriminator_div, *args, **kwargs):
        self.batch_size = batch_size # Defined before super().__init__ as configure_* is called on __init__
        self.annealing_steps = annealing_steps
        self.lr = lr
        self.disc_div = discriminator_div
        self.gan_annealing_steps = discriminator_div * annealing_steps
        self.tprs = []
        self.fprs = []
        self.aucs = []
        super(HumanVAETrainer, self).__init__(*args, **kwargs)
        self.model.to(self.device)

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
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=self.lr),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=self.lr),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=self.lr),
            'discrim': torch.optim.Adam(self.model.expert.discriminator.parameters(), lr=self.lr/self.disc_div)
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
    
    def train_epoch(self, epoch):
        for train_data in self.dataloader:
            self.batch_iteration += 1
            self.train_trace_complete(train_data, epoch)
        
        self.log_discriminator(epoch)

    def train_trace_complete(self, train_data: torch.Tensor, epoch):
        # Zero All Gradients
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()
        self.optimizers['discrim'].zero_grad()

        ## Train Discriminator

        # train discriminator with real data **use l1 & 2
        real_pred = self.model.expert.discriminator(train_data)
        real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
        real_loss.backward()

        # gen fake data
        with torch.no_grad():
            fake_data, _, _ = self.model(train_data)

        # train discriminator with fake data
        fake_pred = self.model.expert.discriminator(fake_data)
        fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        fake_loss.backward()

        self.optimizers['discrim'].step()

        # Forwad Pass Over Entire Model
        x_hat, mu, var = self.model(train_data)
        
        discrim_pred = self.model.expert.discriminator(x_hat)
        gen_loss = F.mse_loss(discrim_pred, torch.ones_like(discrim_pred))
        gan_weight = min(1.0, epoch / self.gan_annealing_steps)

        recon_loss = F.l1_loss(x_hat, train_data.to_dense())
        # Shared VAE Loss
        kl_loss = utils.kl_divergence(mu, var)
        kl_weight = min(1.0, epoch / self.annealing_steps)
        # add generator weight
        loss = (gen_loss * gan_weight) + recon_loss #+ (kl_loss * kl_weight)
        loss.backward()

        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()

        self.writer.add_scalar(f'Loss/discriminator_divisor_{self.disc_div}/KL', kl_loss.item(), self.batch_iteration)
        self.writer.add_scalar(f'Loss/discriminator_divisor_{self.disc_div}/ReconstructionFromTrainingData', loss.item(), self.batch_iteration)
        self.writer.add_scalar(f'Loss/discriminator_divisor_{self.disc_div}/DiscrimRealDataLoss', real_loss.item(), self.batch_iteration)
        self.writer.add_scalar(f'Loss/discriminator_divisor_{self.disc_div}/DiscrimFakeDataLoss', fake_loss.item(), self.batch_iteration)
        self.writer.add_scalar(f'Loss/discriminator_divisor_{self.disc_div}/GeneratorLoss', gen_loss.item(), self.batch_iteration)
    
    def log_discriminator(self, epoch):
        fpr, tpr, auc = get_roc_stats(self.model, self.dataloader)
        
        self.fprs.append(fpr)
        self.tprs.append(tpr)
        self.aucs.append(auc)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        image = PIL.Image.open(buf)
        image_tensor = ToTensor()(image)
        
        self.writer.add_image(f'ROC/discriminator_divisor_{self.disc_div}/EPOCH_{epoch}', image_tensor)
        
        plt.close()