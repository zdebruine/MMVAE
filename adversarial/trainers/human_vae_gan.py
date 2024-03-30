import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
import mmvae.trainers.utils as utils
import adversarial.models.human_vae_gan as HumanVAE
from mmvae.trainers import HPBaseTrainer, BaseTrainerConfig

class HumanVAEGANConfig(BaseTrainerConfig):
    required_hparams = {
        'data_file_path': str,
        'metadata_file_path': str,
        'train_dataset_ratio': float,
        'batch_size': int,
        'shr_vae.optimizer.lr': float,
        'realism_bc.optimizer.lr': float,
        'kl_cyclic.warm_start': int, 
        'kl_cyclic.cycle_length': float, 
        'kl_cyclic.min_beta': float, 
        'kl_cyclic.max_beta': float,
        # 'kl_early_stop_delta': float
    }

class HumanVAEGANTrainer(HPBaseTrainer):
    model: HumanVAE.Model
    
    def __init__(self, device: torch.device, hparams: HumanVAEGANConfig):
        super(HumanVAEGANTrainer, self).__init__(device, hparams)

        #For SSD graphing
        self.kl_list = []
        self.loss_list = []
        self.recon_list = []
        self.md_list = []
        
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
            'shr_vae': torch.optim.Adam(self.model.shared_vae.parameters(), lr=self.hparams['shr_vae.optimizer.lr']),
            'realism_bc': torch.optim.Adam(self.model.realism_bc.parameters(), lr=self.hparams['realism_bc.optimizer.lr']),
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

    def trace_bc_feedback(self, train_data: torch.Tensor, fake_train_data: torch.Tensor):
        real_pred = self.model.realism_bc(train_data)
        real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred), reduction='sum')
        fake_pred = self.model.realism_bc(fake_train_data)
        fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

        total_loss = real_loss + fake_loss

        return total_loss, real_loss, fake_loss
    
    def trace_expert_reconstruction(self, train_data: torch.Tensor):
        x_hat, mu, logvar = self.model(train_data)
        recon_loss = F.l1_loss(x_hat, train_data.to_dense(), reduction='sum')
        kl_loss = utils.kl_divergence(mu, logvar)
        bc_pred = self.model.realism_bc(x_hat)
        bc_loss = F.binary_cross_entropy(bc_pred, torch.ones_like(bc_pred), reduction='sum')
        return x_hat, mu, logvar, recon_loss, kl_loss, bc_loss
    
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
            sum_recon_loss, sum_kl_loss, sum_total_loss, sum_bc_loss = 0, 0, 0, 0
            for test_idx, (test_data, metadata) in enumerate(self.test_loader):
                x_hat, mu, logvar, recon_loss, kl_loss, bc_loss = self.trace_expert_reconstruction(test_data)
                batch_pcc.update(test_data.to_dense(), x_hat)
                recon_loss, kl_loss = recon_loss.item() / test_data.numel(), kl_loss.item() / mu.numel()
                sum_recon_loss += recon_loss 
                sum_kl_loss += kl_loss 
                sum_bc_loss += bc_loss
                sum_total_loss += recon_loss + (kl_weight * kl_loss) + bc_loss

        md = torch.nn.Sequential(
            torch.nn.Linear(60664, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        ).to(self.device)


        fpr, tpr, md_auc = utils.md_eval(md, 1, self.model, self.test_loader)

        self.metrics['Test/Loss/Reconstruction'] = sum_recon_loss / num_batch_samples

        self.metrics['Test/Loss/KL'] = sum_kl_loss / num_batch_samples
        self.metrics['Test/Loss/bc_loss'] = sum_bc_loss / num_batch_samples
        self.metrics['Test/Eval/PCC'] = batch_pcc.compute().item()
        self.metrics['Test/Eval/MD_AUC'] = md_auc
        self.hparams['epochs'] = self.hparams['epochs'] + 1
        if hasattr(self, 'writer'):
            self.writer.add_hparams(dict(self.hparams), self.metrics, run_name=f"{self.hparams['tensorboard.run_name']}_hparams", global_step=epoch)


        self.kl_list.append(sum_kl_loss / num_batch_samples)
        self.recon_list.append(sum_recon_loss / num_batch_samples)
        self.loss_list.append((sum_total_loss / num_batch_samples).cpu())
        self.md_list.append(md_auc)
        
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
            if epoch < warm_start:
                kl_weight = 0
            else:
                kl_weight = (0.15/(self.hparams['epochs']-3))*epoch

            if hasattr(self, 'writer'):
                self.writer.add_scalar('Metric/KLWeight', kl_weight, self.batch_iteration)
            
            with torch.no_grad():
                fake_train_data = self.model(train_data)[0]
            
            fpr, tpr, auc = utils.batch_roc(self.model.realism_bc, train_data, fake_train_data)
            
            if auc < 0.7:
                self.train_trace_bc_feedback(train_data, fake_train_data)

            self.train_trace_expert_reconstruction(train_data, kl_weight)
            self.batch_iteration += 1
            
        self.test_trace_expert_reconstruction(epoch, kl_weight)
        
        for schedular in self.schedulers.values():
            schedular.step()
            
    def train_trace_expert_reconstruction(self, train_data: torch.Tensor, kl_weight: float):

        x_hat, mu, logvar, recon_loss, kl_loss, bc_loss = self.trace_expert_reconstruction(train_data)
        loss: torch.Tensor = recon_loss + (kl_weight * kl_loss) + bc_loss
        self.optimizers['shr_vae'].zero_grad()
        loss.backward()
        self.optimizers['shr_vae'].step()
        
    
    def train_trace_bc_feedback(self, train_data: torch.Tensor, fake_train_data: torch.Tensor):
        total_loss, real_loss, fake_loss = self.trace_bc_feedback(train_data, fake_train_data)
        self.optimizers['realism_bc'].zero_grad()
        total_loss.backward()
        self.optimizers['realism_bc'].step()
    
    def get_run_csv(self, directory: str):
        import pandas as pd
        df = pd.DataFrame({
            'KL': self.kl_list,
            'Reconstruction': self.recon_list,
            'Loss': self.loss_list,
            'MD': self.md_list
        })
        return df.to_csv(directory)