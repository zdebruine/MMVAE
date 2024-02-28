import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
import mmvae.trainers.utils as utils
import mmvae.models.HumanVAE as HumanVAE
from mmvae.trainers.trainer import HPBaseTrainer
import random

class HumanVAETrainer(HPBaseTrainer):
    """
    Trainer class designed for MMVAE model using MutliModalLoader.
    """
    required_hparams = {
        'data_file_path': str,
        'metadata_file_path': str,
        'train_dataset_ratio': float,
        'batch_size': int,
    }
    
    model: HumanVAE.Model
    
    def __init__(self, _device: torch.device, _hparams: dict):
        super(HumanVAETrainer, self).__init__(_device, _hparams, self.required_hparams)
        #self.writer.add_hparams(self.hparams, self.metrics, run_name=self.hparams['tensorboard_run_name'], global_step=-1)
        self.writer.add_text('Model Architecture', str(self.model))
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                             Configuration                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dataloader = None # Overrides and prevents use of BaseTrainer.dataloader
    def __setattr__(self, __name: str, __value: torch.Any) -> None:
        if __name in ('dataloader',):
            raise RuntimeError(f"Use of {__name} is deprecated")
        return super().__setattr__(__name, __value)
    
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
        model = HumanVAE.configure_model().to(self.device)
        return model
        
    def configure_optimizers(self):
        return {
            'encoder': torch.optim.Adam(self.model.expert.encoder.parameters(), lr=self.hparams['encoder_optim_lr']), # weight_decay=l2_reg),
            'decoder': torch.optim.Adam(self.model.expert.decoder.parameters(), lr=self.hparams['decoder_optim_lr']), # weight_decay=l2_reg),
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=self.hparams['shr_vae_optim_lr']), #  weight_decay=l2_reg)
        }
        
    def configure_schedulers(self) -> dict[str, LRScheduler]:
        return { 
                key: torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=self.hparams[f'{key}_optim_step_size'], 
                    gamma=self.hparams[f"{key}_optim_gamma"])
                for key, optimizer in self.optimizers.items()
            }
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Trace Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def trace_expert_reconstruction(self, train_data: torch.Tensor):
        batch_size = train_data.shape[0]
        x_hat, mu, logvar = self.model(train_data)
        recon_loss = F.mse_loss(x_hat, train_data.to_dense(), reduction='sum') / batch_size
        kl_loss = utils.kl_divergence(mu, logvar) / batch_size
        return x_hat, mu, logvar, recon_loss, kl_loss
    
    def test_trace_expert_reconstruction(self, epoch, kl_weight):
        self.metrics['Test/Loss/ReconstructionLoss'] = 0.0
        self.metrics['Test/Loss/KL'] = 0.0
        self.metrics['Test/Loss/Total'] = 0.0
        self.metrics['Test/Loss/NonZeroFeatureReconstruction'] = 0.0
        self.metrics['Test/Loss/ZeroFeatureReconstruction'] = 0.0
        self.metrics['Test/Eval/PCC'] = 0.0
    
        with torch.no_grad():
            self.model.eval()
            num_samples = len(self.test_loader)
            batch_pcc = utils.BatchPCC()
            for i, (test_data, metadata) in enumerate(self.test_loader):
                x_hat, mu, logvar, recon_loss, kl_loss = self.trace_expert_reconstruction(test_data)
                dense_test_data = test_data.to_dense()
                non_zero_mask = test_data.to_dense() != 0
                batch_size = test_data.shape[0]
                self.metrics['Test/Loss/NonZeroFeatureReconstruction'] += F.mse_loss(x_hat[non_zero_mask], dense_test_data[non_zero_mask], reduction='sum') / batch_size
                zero_mask = ~non_zero_mask
                self.metrics['Test/Loss/ZeroFeatureReconstruction'] += F.mse_loss(x_hat[zero_mask], dense_test_data[zero_mask], reduction='sum') / batch_size

                if i == -1: # TODO: import matplotlib
                    random_image_idx = random.randint(0, len(test_data) - 1)
                    utils.save_image(test_data[random_image_idx], '/home/denhofja/real_cell_image.png')
                    utils.save_image(x_hat[random_image_idx], '/home/denhofja/x_hat.png')
                    
                batch_pcc.update(dense_test_data, x_hat)
                
                recon_loss, kl_loss = recon_loss.item() / num_samples, kl_loss.item() / num_samples
                self.metrics['Test/Loss/ReconstructionLoss'] += recon_loss
                self.metrics['Test/Loss/KL'] += kl_loss
                self.metrics['Test/Loss/Total'] += recon_loss + (kl_weight * kl_loss)
                
        self.metrics['Test/Eval/PCC'] = batch_pcc.compute().item()
        # for metric in self.metrics:
        #     self.writer.add_scalar(metric, self.metrics[metric], global_step=epoch)
        self.hparams['epochs'] = self.hparams['epochs'] + 1
        self.writer.add_hparams(self.hparams, self.metrics, run_name=f"{self.hparams['tensorboard_run_name']}_hparams", global_step=epoch)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          Train Configuration                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def train(self, epochs, load_snapshot=False):
        self.batch_iteration = 0
        super().train(epochs, load_snapshot)
    
    def train_epoch(self, epoch):
        num_samples = len(self.train_loader)
        kl_weight = 0.5
        warm_start = self.hparams['kl_cyclic_warm_start']
        for (train_data, metadata) in self.train_loader:
            kl_weight = utils.cyclic_annealing((self.batch_iteration - (warm_start * num_samples)), self.hparams['kl_cyclic_cycle_length'], min_beta=self.hparams['kl_cyclic_min_beta'], max_beta=self.hparams['kl_cyclic_max_beta'])
            kl_weight = 0 if epoch < warm_start else kl_weight
            self.writer.add_scalar('Metric/KLWeight', kl_weight, self.batch_iteration)
            self.train_trace_expert_reconstruction(train_data, kl_weight)
            self.batch_iteration += 1
            
        self.test_trace_expert_reconstruction(epoch, kl_weight)
        
        # for schedular in self.schedulers.values():
        #     schedular.step()
            
    def train_trace_expert_reconstruction(self, train_data: torch.Tensor, kl_weight: float):

        x_hat, mu, logvar, recon_loss, kl_loss = self.trace_expert_reconstruction(train_data)
        loss: torch.Tensor = recon_loss + (kl_weight * kl_loss)
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['encoder'].zero_grad()
        self.optimizers['decoder'].zero_grad()
        
        loss.backward()
        
        # if self.batch_iteration % 100 == 0:
        #     for name, parameter in self.model.named_parameters():
        #         if parameter.requires_grad and parameter.grad is not None:
        #             self.writer.add_histogram(f"Gradients/{name}_{str(parameter.grad.shape)}", parameter.grad, self.batch_iteration)
        
        self.optimizers['shr_vae'].step()
        self.optimizers['encoder'].step()
        self.optimizers['decoder'].step()