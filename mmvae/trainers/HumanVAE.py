import torch
import torch.nn.functional as F
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import CellCensusDataLoader
from datetime import datetime

class HumanVAETrainer(BaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """

    model: HumanVAE.Model
    dataloader: CellCensusDataLoader

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size # Defined before super().__init__ as configure_* is called on __init__
        super(HumanVAETrainer, self).__init__(*args, **kwargs)
        self.model.to(self.device)

    def configure_dataloader(self):
        return CellCensusDataLoader(
            'human', 
            directory_path="/active/debruinz_project/CellCensus_3M/",
            masks=['*human_chunk*'], 
            batch_size=self.batch_size, 
            num_workers=3
        )

    def configure_model(self):
        return HumanVAE.configure_model()
    
    def configure_optimizers(self):
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=0.001, weight_decay=1e-5),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=0.001, weight_decay=1e-5),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=.0001)
        }
    
    def configure_schedulers(self):
        from torch.optim.lr_scheduler import StepLR
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

    def train_epoch(self, epoch):
        for iteration, (data, _) in enumerate(self.dataloader):
            train_data = data.to(self.device)
    
            self.train_trace_expert_autoencoder(iteration, train_data)
                
            # self.train_trace_complete(iteration, train_data)

    def train_trace_expert_autoencoder(self, iteration: int, train_data: torch.Tensor):
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()
    
        x = self.model.expert(train_data)
        
        loss = F.l1_loss(x, train_data.to_dense())
        loss.backward()

        self.writer.add_scalar('Expert_Loss/Reconstruction Loss (l1_loss)', loss.item(), iteration)

        for model in ('encoder', 'decoder'):
            self.optimizers[model].step()
            self.schedulers[model].step()
            


    def train_trace_complete(self, iteration: int, train_data: torch.Tensor):
        # Zero All Gradients
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()

        # Forwad Pass Over Entire Model
        vae_input, vae_output, mu, var, shared_encoder_outputs, shared_decoder_outputs, expert_output = self.model(train_data)
        # Expert Reconstruction Loss
        expert_recon_loss = F.l1_loss(expert_output, train_data.to_dense())
        # Shared VAE Loss
        vae_loss, kl_loss = self.vae_loss(expert_output, train_data.to_dense(), mu, var)
        loss = (vae_loss * 1.2) + kl_loss 
        loss.backward()
        #loss = expert_recon_loss + vae_loss + (kl_loss * 0.1)
        #loss.backward()
    
        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()

        #self.writer.add_scalar('Loss/Expert_Recon', expert_recon_loss.item(), iteration)
        self.writer.add_scalar('Loss/VAE', vae_loss.item(), iteration)
        self.writer.add_scalar('Loss/KL', kl_loss.item(), iteration)
        self.writer.add_scalar('Loss/Total', loss.item(), iteration)
        self.writer.flush()

def main(device, batch_size, log_name):
    # Define any hyperparameters
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create trainer instance
    trainer = HumanVAETrainer(
        batch_size,
        device,
        log_dir=f"/home/denhofja/logs/{log_name}"
    )
    # Train model with number of epochs
    trainer.train(epochs=1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Example application')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('log_name', type=str)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device, args.batch_size, args.log_name)

    