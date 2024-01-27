import torch
import torch.nn.functional as F
import d_mmvae.trainers.utils as utils
from d_mmvae.models.MMVAE import MMVAE
from d_mmvae.trainers.trainer import BaseTrainer
from d_mmvae.data import MultiModalLoader

class MMVAETrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: MMVAE
    dataloader: MultiModalLoader

    def __init__(self, *args, **kwargs):
        super(MMVAETrainer, self).__init__(*args, **kwargs)
        self.model.to(self.device)
        
        self.expert_class_indices = [i for i in range(len(self.model.experts)) ]
    
    def vae_loss(self, predicted, target, mu, sigma, kl_weight = 1):
        vae_recon_loss = F.mse_loss(predicted, target)
        kl_loss = utils.kl_divergence(mu, sigma)
        return vae_recon_loss + kl_loss * kl_weight
    
    def expert_discriminator_loss(self, data, expert, real=True):
        output = self.model.experts[expert].discriminator(data)
        labels = torch.ones(output.size(), device=self.device) if real else torch.zeros(output.size(), device=self.device)
        loss = F.binary_cross_entropy(output, labels)
        return loss

    def train_epoch(self):
        for iteration, (data, expert) in enumerate(self.dataloader):

            self.model.set_expert(expert)
            train_data = data.to(self.device)
            
            # Forwad Pass Over Entire Model
            shared_input, mu, sigma, shared_encoder_outputs, shared_decoder_outputs, shared_output, expert_output = self.model(train_data)

            # Train Expert Discriminator
            if iteration % 5 == 0:
                # Zero Expert Discriminator Gradients
                opt_exp_disc = self.optimizers[f'{expert}-disc']
                opt_exp_disc.zero_grad()
                # Train discriminator on Real Data
                real_loss = self.expert_discriminator_loss(train_data, expert)
                real_loss.backward()
                # Train discriminator on fake data
                fake_loss = self.expert_discriminator_loss(expert_output.detach(), expert, real=False)
                # Update Expert Discriminator Gradients
                fake_loss.backward()
                opt_exp_disc.step()

            # Zero Shared VAE and Expert Gradients
            self.optimizers['shr_vae'].zero_grad()
            self.optimizers['shr_enc_disc'].zero_grad()
            # Expert Reconstruction Loss
            expert_recon_loss = F.mse_loss(expert_output, train_data.to_dense())
            # Shared VAE Loss
            vae_loss = self.vae_loss(shared_output, shared_input, mu, sigma)
            # Shared Encoder Adverserial Feedback
            labels = torch.tensor([self.expert_class_indices] * 32, dtype=float, device=self.device)
            shr_enc_adversial_loss = F.cross_entropy(shared_encoder_outputs, labels)
            # Shared Expert Discriminator Loss
            shared_loss = self.expert_discriminator_loss(expert_output.detach(), expert, real=False)
            adv_weight = -0.1 # TODO: Add annealing factor
            total_loss = expert_recon_loss + vae_loss + adv_weight * shr_enc_adversial_loss + shared_loss
            total_loss.backward()
        
            self.optimizers['shr_enc_disc'].step()
            self.optimizers['shr_vae'].step()
            self.optimizers[f'{expert}-enc'].step()
            self.optimizers[f'{expert}-dec'].step()