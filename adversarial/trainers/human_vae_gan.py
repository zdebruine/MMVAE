import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
import mmvae.trainers.utils as utils
import adversarial.models.human_vae_gan as HumanVAE
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

    def train_bc_batch(self, train_data, fake_batch_data):
        self.optimizers['realism_bc'].zero_grad()

        real_pred = self.model.realism_bc(train_data)
        real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred), reduction='sum')
        
        fake_pred = self.model.realism_bc(fake_batch_data)
        fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred), reduction='sum')

        D_loss = real_loss + fake_loss    
        D_loss.backward()

        self.optimizers['realism_bc'].step()

        return D_loss, real_loss, fake_loss
    
    def train_vae_batch(self, train_data, kl_weight):
        self.optimizers['shr_vae'].zero_grad()
        self.optimizers['expert.encoder'].zero_grad()
        self.optimizers['expert.decoder'].zero_grad()
        G_loss = torch.Tensor

        x_hat, mu, logvar = self.model(train_data)
        #score = self.model.realism_bc(x_hat)
        recon_loss = F.mse_loss(x_hat, train_data.to_dense())
        kl_loss = utils.kl_divergence(mu, logvar, reduction='mean')
        #G_loss = F.mse_loss(score, torch.ones_like(score), reduction='sum')

        loss: torch.Tensor = recon_loss + (kl_weight * kl_loss) #+ G_loss
        loss.backward()

        self.optimizers['shr_vae'].step()
        self.optimizers['expert.encoder'].step()
        self.optimizers['expert.decoder'].step()

        return x_hat, mu, logvar, recon_loss, kl_loss, G_loss
        
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
            sum_recon_loss, sum_kl_loss, sum_total_loss, sum_G_loss, sum_D_loss = 0, 0, 0, 0, 0

            for test_idx, (test_data, metadata) in enumerate(self.test_loader):
                # VAE
                x_hat, mu, logvar = self.model(test_data)
                score = self.model.realism_bc(x_hat)
                recon_loss = F.mse_loss(x_hat, test_data.to_dense())
                kl_loss = utils.kl_divergence(mu, logvar, reduction='mean')
                #G_loss = F.mse_loss(score, torch.ones_like(score), reduction='sum')
                # end VAE

                # Discriminator
                # real_pred = self.model.realism_bc(test_data)
                # real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred), reduction='sum')
                
                # fake_pred = self.model.realism_bc(x_hat)
                # fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred), reduction='sum')

                # D_loss = real_loss + fake_loss   
                # End Discriminator

                #batch_pcc.update(test_data.to_dense(), x_hat)

                recon_loss, kl_loss = recon_loss.item() / test_data.numel(), kl_loss.item() / mu.numel()
                sum_recon_loss += recon_loss 
                sum_kl_loss += kl_loss 
                #sum_G_loss += G_loss
                #sum_D_loss += D_loss
                sum_total_loss += recon_loss + (kl_weight * kl_loss) #+ G_loss
        
        md = torch.nn.Sequential(
                torch.nn.Linear(60664, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()
            ).to(self.device)

        fpr, tpr, md_auc = utils.md_eval(md=md, md_epochs=1, gen=self.model, dataloader=self.test_loader)

        self.metrics['Test/Loss/Reconstruction'] = sum_recon_loss / num_batch_samples
        self.metrics['Test/Loss/KL'] = sum_kl_loss / num_batch_samples
        self.metrics['Test/Loss/Total_loss'] = sum_total_loss / num_batch_samples
        #self.metrics['Test/Loss/G_Loss']= sum_G_loss / num_batch_samples
        #self.metrics['Test/Loss/D_Loss']= sum_D_loss / num_batch_samples
        
        #self.metrics['Test/Eval/PCC'] = batch_pcc.compute().item()
        self.metrics['Test/Eval/MD_AUC'] = md_auc
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
            #kl_weight = utils.cyclic_annealing((self.batch_iteration - (warm_start * num_batch_samples)), cycle_length, min_beta=self.hparams['kl_cyclic.min_beta'], max_beta=self.hparams['kl_cyclic.max_beta'])
            #kl_weight = 0 if epoch < warm_start else kl_weight
            kl_weight = 0.5

            with torch.no_grad():
                fake_batch_data, _, _ = self.model(train_data)
            
            #batch_fpr, batch_tpr, batch_auc = utils.batch_roc(self.model.realism_bc, train_data, fake_batch_data)

            #D_loss, real_loss, fake_loss = self.train_bc_batch(train_data, fake_batch_data)
            x_hat, mu, logvar, recon_loss, kl_loss, G_loss = self.train_vae_batch(train_data, kl_weight)

            if hasattr(self, 'writer'):
                self.writer.add_scalar('Batch_Scale/KLWeight', kl_weight, self.batch_iteration)
                #self.writer.add_scalar('Batch_Scale/BC_AUC', batch_auc, self.batch_iteration)
                #self.writer.add_scalar('Batch_Scale/D_Loss', D_loss.item(), self.batch_iteration)
                #self.writer.add_scalar('Batch_Scale/G_Loss', G_loss.item(), self.batch_iteration)
                self.writer.add_scalar('Batch_Scale/recon_loss', recon_loss.item(), self.batch_iteration)
                self.writer.add_scalar('Batch_Scale/kl_loss', kl_loss.item(), self.batch_iteration)
            
            self.batch_iteration += 1

        #self.writer.add_scalar('Epoch_Scale/BC_AUC', batch_auc, self.batch_iteration)
        #self.writer.add_scalar('Epoch_Scale/D_Loss', D_loss.item(), self.batch_iteration)
        # self.writer.add_scalar('Epoch_Scale/G_Loss', G_loss.item(), self.batch_iteration)
        # self.writer.add_scalar('Epoch_Scale/recon_loss', recon_loss.item(), self.batch_iteration)
        # self.writer.add_scalar('Epoch_Scale/kl_loss', kl_loss.item(), self.batch_iteration)
        self.test_trace_expert_reconstruction(epoch, kl_weight)

        # for schedular in self.schedulers.values():
        #     schedular.step()
            
