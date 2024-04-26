import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers import HPBaseTrainer, BaseTrainerConfig

class HumanVAEConfig(BaseTrainerConfig):
    required_hparams = {
        'data_file_path': str,
        'metadata_file_path': str,
        'train_dataset_ratio': float,
        'batch_size': int,
        'expert.encoder.optimizer.lr': float,
        'expert.decoder.optimizer.lr': float, 
        'shr_vae.optimizer.lr': float, 
        'kl_cyclic.warm_start': int, 
        'kl_cyclic.cycle_length': float, 
        'kl_cyclic.min_beta': float, 
        'kl_cyclic.max_beta': float 
    }

class HumanVAETrainer(HPBaseTrainer):
    
    model: HumanVAE.Model
    
    def __init__(self, device: torch.device, hparams: HumanVAEConfig):
        super(HumanVAETrainer, self).__init__(device, hparams)
        if hasattr(self, 'writer'):
            self.writer.add_text('Model Architecture', str(self.model))
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                             Configuration                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def configure_dataloader(self):
        from mmvae.data import configure_singlechunk_dataloaders
        self.train_loader, self.test_loader = configure_singlechunk_dataloaders(
            data_file_path=self.hparams['data_file_path'],
            metadata_file_path=self.hparams['metadata_file_path'],
            train_ratio=self.hparams['train_dataset_ratio'],
            batch_size=self.hparams['batch_size'],
            device=self.device
        )
        
    def configure_model(self) -> Module:
        model = HumanVAE.configure_model(self.hparams.config)
        return model.to(self.device)
        
    def configure_optimizers(self):
        return {
            'expert.encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=self.hparams['expert.encoder.optimizer.lr']),
            'expert.decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=self.hparams['expert.decoder.optimizer.lr']),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=self.hparams['shr_vae.optimizer.lr']),
        }
        
    def configure_schedulers(self) -> dict[str, LRScheduler]:
        return { 
                key: torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=self.hparams[f'{key}.optimizer.schedular.step_size'], 
                    gamma=self.hparams[f"{key}.optimizer.schedular.gamma"])
                for key, optimizer in self.optimizers.items() 
                if f'{key}.optimizer.schedular.step_size' in self.hparams 
                    and self.hparams[f'{key}.optimizer.schedular.step_size'] != "" 
                    and f'{key}.optimizer.schedular.gamma' in self.hparams
                    and self.hparams[f'{key}.optimizer.schedular.gamma'] != ""
            }
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Trace Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def trace_expert_reconstruction(self, train_data: torch.Tensor):
        x_hat, mu, logvar = self.model(train_data)
        recon_loss = F.mse_loss(x_hat, train_data.to_dense(), reduction='sum')
        kl_loss = utils.kl_divergence(mu, logvar)
        return x_hat, mu, logvar, recon_loss, kl_loss
    
    def log_non_zero_and_zero_reconstruction(self, inputs, targets):
        non_zero_mask = inputs != 0
        self.metrics['Test/Loss/NonZeroFeatureReconstruction'] += F.mse_loss(inputs[non_zero_mask], targets[non_zero_mask], reduction='sum') 
        zero_mask = ~non_zero_mask
        self.metrics['Test/Loss/ZeroFeatureReconstruction'] += F.mse_loss(inputs[zero_mask], targets[zero_mask], reduction='sum') 
    
    def test_trace_expert_reconstruction(self, epoch, kl_weight):
        with torch.no_grad():
            self.model.eval()
            num_batch_samples = len(self.test_loader)
            batch_pcc = utils.BatchPCC()
            sum_recon_loss, sum_kl_loss, sum_total_loss = 0, 0, 0
            for test_idx, (test_data, metadata) in enumerate(self.test_loader):
                x_hat, mu, logvar, recon_loss, kl_loss = self.trace_expert_reconstruction(test_data)
                batch_pcc.update(test_data.to_dense(), x_hat)
                recon_loss, kl_loss = recon_loss.item() / test_data.numel(), kl_loss.item() / mu.numel()
                sum_recon_loss += recon_loss 
                sum_kl_loss += kl_loss 
                sum_total_loss += recon_loss + (kl_weight * kl_loss)
        
        self.metrics['Test/Loss/Reconstruction'] = sum_recon_loss / num_batch_samples
        self.metrics['Test/Loss/KL'] = sum_kl_loss / num_batch_samples
        self.metrics['Test/Eval/PCC'] = batch_pcc.compute().item()
        self.hparams['epochs'] = self.hparams['epochs'] + 1
        if hasattr(self, 'writer'):
            self.writer.add_hparams(dict(self.hparams), self.metrics, run_name=f"{self.hparams['tensorboard.run_name']}_hparams", global_step=epoch)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Train Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def train(self, epochs, load_snapshot=False):
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)

    def train_epoch(self, epoch):
        self.model.train(True) # Ensure model is in train mode after testing
        num_batch_samples = len(self.train_loader)
        warm_start = self.hparams['kl_cyclic.warm_start']
        cycle_length = len(self.train_loader) if self.hparams['kl_cyclic.cycle_length'] == "length_of_dataset" else self.hparams['kl_cyclic.cycle_length']
        for (train_data, metadata) in self.train_loader:
            kl_weight = utils.cyclic_annealing((self.batch_iteration - (warm_start * num_batch_samples)), cycle_length, min_beta=self.hparams['kl_cyclic.min_beta'], max_beta=self.hparams['kl_cyclic.max_beta'])
            kl_weight = 0 if epoch < warm_start else kl_weight
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Metric/KLWeight', kl_weight, self.batch_iteration)
            self.train_trace_expert_reconstruction(train_data, kl_weight)
            self.batch_iteration += 1
            
        self.test_trace_expert_reconstruction(epoch, kl_weight)
        
        for schedular in self.schedulers.values():
            schedular.step()
            
    def train_trace_expert_reconstruction(self, train_data: torch.Tensor, kl_weight: float):

        x_hat, mu, logvar, recon_loss, kl_loss = self.trace_expert_reconstruction(train_data)
        loss: torch.Tensor = recon_loss + (kl_weight * kl_loss)
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['expert.encoder'].zero_grad()
        self.optimizers['expert.decoder'].zero_grad()
        loss.backward()
        self.optimizers['shr_vae'].step()
        self.optimizers['expert.encoder'].step()
        self.optimizers['expert.decoder'].step()
