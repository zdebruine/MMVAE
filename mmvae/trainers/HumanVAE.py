import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers.trainer import HPBaseTrainer

class HumanVAETrainer(HPBaseTrainer):
    
    required_hparams = {
        'data_file_path': str,
        'metadata_file_path': str,
        'train_dataset_ratio': float,
        'batch_size': int,
    }
    
    model: HumanVAE.Model
    
    def __init__(self, _device: torch.device, _hparams: dict):
        self._initial_hparams = _hparams
        super(HumanVAETrainer, self).__init__(_device, _hparams, self.required_hparams)
        self.writer.add_text('Model Architecture', str(self.model))
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                             Configuration                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def configure_dataloader(self):
        import mmvae.data.utils as utils
        (train_data, train_metadata), (validation_data, validation_metadata) = utils.split_data_and_metadata(
            data_file_path=self.hparams['data_file_path'],
            metadata_file_path=self.hparams['metadata_file_path'],
            train_ratio=self.hparams['train_dataset_ratio'])
        
        from mmvae.data.datasets.CellCensusDataSet import CellCensusDataset, collate_fn
        train_dataset = CellCensusDataset(train_data.to(self.device), train_metadata)
        test_dataset = CellCensusDataset(validation_data.to(self.device), validation_metadata)
        
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=self.hparams['batch_size'],
            collate_fn=collate_fn,
        )
        
    def configure_model(self) -> Module:
        model = HumanVAE.configure_model(self._initial_hparams).to(self.device)
        return model
        
    def configure_optimizers(self):
        return {
            'expert_encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=self.hparams['expert_encoder_optim_lr']),
            'expert_decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=self.hparams['expert_decoder_optim_lr']),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=self.hparams['shr_vae_optim_lr']),
        }
        
    def configure_schedulers(self) -> dict[str, LRScheduler]:
        return { 
                key: torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=self.hparams[f'{key}_optim_sched_step_size'], 
                    gamma=self.hparams[f"{key}_optim_sched_gamma"])
                for key, optimizer in self.optimizers.items() 
                if f'{key}_optim_sched_step_size' in self.hparams \
                    and self.hparams[f'{key}_optim_sched_step_size'] != "" 
                    and f'{key}_optim_sched_gamma' in self.hparams
                    and self.hparams[f'{key}_optim_sched_gamma'] != ""
            }
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Trace Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def trace_expert_reconstruction(self, train_data: torch.Tensor):
        x_hat, mu, logvar = self.model(train_data)
        recon_loss = F.mse_loss(x_hat, train_data.to_dense(), reduction='mean')
        kl_loss = utils.kl_divergence(mu, logvar)
        return x_hat, mu, logvar, recon_loss, kl_loss
    
    def log_non_zero_and_zero_reconstruction(self, inputs, targets):
        non_zero_mask = inputs != 0
        self.metrics['Test/Loss/NonZeroFeatureReconstruction'] += F.mse_loss(inputs[non_zero_mask], targets[non_zero_mask], reduction='mean') 
        zero_mask = ~non_zero_mask
        self.metrics['Test/Loss/ZeroFeatureReconstruction'] += F.mse_loss(inputs[zero_mask], targets[zero_mask], reduction='mean') 
    
    def test_trace_expert_reconstruction(self, epoch, kl_weight):
        with torch.no_grad():
            self.model.eval()
            num_batch_samples = len(self.test_loader)
            batch_pcc = utils.BatchPCC()
            sum_recon_loss, sum_kl_loss, sum_total_loss = 0, 0, 0
            for test_idx, (test_data, metadata) in enumerate(self.test_loader):
                x_hat, mu, logvar, recon_loss, kl_loss = self.trace_expert_reconstruction(test_data)
                batch_pcc.update(test_data.to_dense(), x_hat)
                recon_loss, kl_loss = recon_loss.item(), kl_loss.item()
                sum_recon_loss += recon_loss
                sum_kl_loss += kl_loss
                sum_total_loss += recon_loss + (kl_weight * kl_loss)
        
        self.metrics['Test/Loss/Reconstruction'] = sum_recon_loss / num_batch_samples
        self.metrics['Test/Loss/KL'] = sum_kl_loss / num_batch_samples
        self.metrics['Test/Eval/PCC'] = batch_pcc.compute().item()
        self.hparams['epochs'] = self.hparams['epochs'] + 1
        self.writer.add_hparams(self.hparams, self.metrics, run_name=f"{self.hparams['tensorboard_run_name']}_hparams", global_step=epoch)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Train Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def train(self, epochs, load_snapshot=False):
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)
    
    def train_epoch(self, epoch):
        num_batch_samples = len(self.train_loader)
        kl_weight = 0.5
        warm_start = self.hparams['kl_cyclic_warm_start']
        cycle_length = len(self.train_loader) if self.hparams['kl_cyclic_cycle_length'] == "length_of_dataset" else self.hparams['kl_cyclic_cycle_length']
        for (train_data, metadata) in self.train_loader:
            kl_weight = utils.cyclic_annealing((self.batch_iteration - (warm_start * num_batch_samples)), cycle_length, min_beta=self.hparams['kl_cyclic_min_beta'], max_beta=self.hparams['kl_cyclic_max_beta'])
            kl_weight = 0 if epoch < warm_start else kl_weight
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
        self.optimizers['expert_encoder'].zero_grad()
        self.optimizers['expert_decoder'].zero_grad()
        loss.backward()
        self.optimizers['shr_vae'].step()
        self.optimizers['expert_encoder'].step()
        self.optimizers['expert_decoder'].step()