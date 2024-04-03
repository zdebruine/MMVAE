import torch
import torch.nn.functional as F
import mmvae.trainers.utils as utils
import torch.utils.tensorboard as tb
import torch.nn as nn
import mmvae.models.Arch_Model as Arch_Model
from mmvae.data import configure_singlechunk_dataloaders

class VAETrainer:

    def __init__(self, device, model=Arch_Model.VAE(), batch_size=256, learning_rate=0.00001, num_epochs=20, start_kl=0.0, end_kl=0.15, annealing_start=3, annealing_steps=17):
        #Configure
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        #Hyperparameters
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.start_kl = start_kl
        self.end_kl = end_kl
        self.annealing_start = annealing_start
        self.annealing_steps = annealing_steps
        self.batch_size = batch_size
        #Tensorboard
        self.writer = tb.SummaryWriter()
        #Load Data
        self.train_loader = configure_singlechunk_dataloaders(
            data_file_path='/active/debruinz_project/CellCensus_3M_Full/3m_human_full.npz',
            metadata_file_path=None,
            train_ratio=1,
            batch_size=self.batch_size,
            device=None
        )

    def loss_function(self, recon_x, x: torch.Tensor, mu, logvar):
        reconstruction_loss = F.l1_loss(recon_x, x.to_dense(), reduction='sum') / self.batch_size
        # kl_divergence = utils.kl_divergence(mu, logvar, reduction='sum') / self.batch_size
        # kl_divergence = kl_divergence.mean()
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / self.batch_size
        return reconstruction_loss, kl_divergence

    def train(self):
        print("Start Training ....")
        for epoch in range(self.num_epochs):
            for i, (x, _) in enumerate(self.train_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(x)

                #Check starting epoch for kl
                annealing = 0
                if epoch >= self.annealing_start:
                    annealing_ratio = (epoch - self.annealing_start) / self.annealing_steps
                    annealing = self.start_kl + annealing_ratio * (self.end_kl - self.start_kl)
                

                recon_loss, kl_loss = self.loss_function(recon_batch, x, mu, logvar)
                annealing_kl = kl_loss * annealing
                loss = recon_loss + annealing_kl

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()


                self.writer.add_scalar('Loss/Iteration', loss.item(), epoch * len(self.train_loader) + i)

                if (i + 1) % 3125 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, self.num_epochs, i + 1, len(self.train_loader), loss.item()))

            #Write loss to tensorboard  
            self.writer.add_scalar('Annealing Schedule', annealing, epoch)
            self.writer.add_scalar('Loss/KL', kl_loss.item(), epoch)
            self.writer.add_scalar('Loss/L1', recon_loss.item(), epoch)
            self.writer.add_scalar('Loss/Total', loss.item(), epoch)
    
        self.writer.flush()