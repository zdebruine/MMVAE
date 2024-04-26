import torch
import torch.nn.functional as F
import mmvae.trainers.utils as utils
import mmvae.models.ExampleModel as ExampleModel
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MultiModalLoader, ChunkedCellCensusDataLoader


class ExampleTrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: ExampleModel.Model
    dataloader: MultiModalLoader

    def __init__(self, batch_size, *args, **kwargs):
        # Defined before super().__init__ as configure_* is called on __init__
        self.batch_size = batch_size
        super(ExampleTrainer, self).__init__(*args, **kwargs)
        self.model.to(self.device)
        self.expert_class_indices = [i for i in range(len(self.model.experts))]

    def configure_dataloader(self):
        expert1 = ChunkedCellCensusDataLoader('expert1', directory_path="/active/debruinz_project/tony_boos/csr_chunks", masks=[
                                              'chunk*'], batch_size=self.batch_size, num_workers=2)
        expert2 = ChunkedCellCensusDataLoader('expert2', directory_path="/active/debruinz_project/tony_boos/csr_chunks", masks=[
                                              'chunk*'], batch_size=self.batch_size, num_workers=2)
        return MultiModalLoader(expert1, expert2)

    def configure_model(self):
        return ExampleModel.configure_model(num_experts=2)

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizers = {}
        for name in self.model.experts.keys():
            expert = self.model.experts[name]
            optimizers[f'{name}-enc'] = torch.optim.Adam(expert.encoder.parameters())
            optimizers[f'{name}-dec'] = torch.optim.Adam(expert.decoder.parameters())
            optimizers[f'{name}-disc'] = torch.optim.Adam(expert.discriminator.parameters())

        optimizers['shr_enc_disc'] = torch.optim.Adam(
            self.model.shared_vae.encoder.discriminator.parameters())
        optimizers['shr_vae'] = torch.optim.Adam(list(self.model.shared_vae.encoder.parameters(
        )) + list(self.model.shared_vae.decoder.parameters()))
        return optimizers

    def vae_loss(self, predicted, target, mu, sigma, kl_weight=1):
        # sum over the input dimensions then mean over the batch
        vae_recon_loss = F.mse_loss(
            predicted, target, reduction=None).sum(dim=1).mean()
        kl_loss = utils.kl_divergence(mu, sigma, reduction="mean")
        return vae_recon_loss + kl_loss * kl_weight

    def expert_discriminator_loss(self, data, expert, real=True):
        output = self.model.experts[expert].discriminator(data)
        labels = torch.ones(output.size(), device=self.device) if real else torch.zeros(
            output.size(), device=self.device)
        loss = F.binary_cross_entropy(
            output, labels)
        return loss

    def train_epoch(self, epoch):  

        for iteration, (data, expert) in enumerate(self.dataloader):
            print("Starting Iteration", iteration, flush=True)
            self.model.set_expert(expert)
            train_data = data.to(self.device)

            # Forwad Pass Over Entire Model
            shared_input, shared_output, mu, var, shared_encoder_outputs, shared_decoder_outputs, expert_output = self.model(
                train_data)

            # Train Expert Discriminator
            if iteration % 5 == 0:
                # Zero Expert Discriminator Gradients
                opt_exp_disc = self.optimizers[f'{expert}-disc']
                opt_exp_disc.zero_grad()
                # Train discriminator on Real Data
                real_loss = self.expert_discriminator_loss(train_data, expert)
                real_loss.backward()
                # Train discriminator on fake data
                fake_loss = self.expert_discriminator_loss(
                    expert_output.detach(), expert, real=False)
                # Update Expert Discriminator Gradients
                fake_loss.backward()
                opt_exp_disc.step()

            # Zero Shared VAE and Expert Gradients
            self.optimizers['shr_vae'].zero_grad()
            self.optimizers['shr_enc_disc'].zero_grad()
            # Expert Reconstruction Loss
            expert_recon_loss = F.mse_loss(
                expert_output, train_data.to_dense())
            # Shared VAE Loss
            vae_recon_loss = F.mse_loss(shared_output, shared_input) #MSE
            kl_loss = utils.kl_divergence(mu, var) #KL
            #combined vae loss 
            vae_loss = self.vae_loss(shared_output, shared_input, mu, var)
            # Shared Encoder Adverserial Feedback
            labels = torch.tensor(
                [self.expert_class_indices] * 32, dtype=float, device=self.device)
            shr_enc_adversial_loss = F.cross_entropy(
                shared_encoder_outputs, labels)
            # Shared Expert Discriminator Loss
            shared_loss = self.expert_discriminator_loss(
                expert_output.detach(), expert, real=False)
            adv_weight = -0.1  # TODO: Add annealing factor
            total_loss = expert_recon_loss + vae_loss + \
                adv_weight * shr_enc_adversial_loss + shared_loss
            total_loss.backward()

            self.optimizers['shr_enc_disc'].step()
            self.optimizers['shr_vae'].step()
            self.optimizers[f'{expert}-enc'].step()
            self.optimizers[f'{expert}-dec'].step()     

            self.writer.add_scalar('Loss/Expert_Recon',
                                   expert_recon_loss.item(), iteration)
            self.writer.add_scalar('Loss/VAE', vae_loss.item(), iteration)
            self.writer.add_scalar(
                'Loss/Shared_Encoder_Adversarial', shr_enc_adversial_loss.item(), iteration)
            self.writer.add_scalar(
                'Loss/Shared', shared_loss.item(), iteration)
            self.writer.add_scalar('Loss/Total', total_loss.item(), iteration)
            self.writer.flush()
