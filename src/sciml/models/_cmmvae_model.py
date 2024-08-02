import torch

from ._base_model import BaseModel
from sciml.modules import CMMVAE
from sciml.constants import REGISTRY_KEYS as RK
from lightning.pytorch.callbacks import ModelCheckpoint


class CMMVAEModel(BaseModel):
    """
    Multi-Modal Variational Autoencoder (MMVAE) model for handling expert-specific data.

    Args:
        module (MMVAE): Multi-Modal VAE module.
        **kwargs: Additional keyword arguments for the base VAE model.

    Attributes:
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
    """
    
    module: CMMVAE
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # Disable automatic optimization for manual control
        
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): Batch of data containing inputs and target labels.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        x, metadata, expert_id = batch
        # Retrieve optimizers
        vae_optimizer, *expert_optimizers = self.optimizers()
        expert_optimizer = expert_optimizers[self.optimizer_map[expert_id] - 1]
        # Zero the gradients for the shared and expert-specific optimizers
        vae_optimizer.zero_grad()
        expert_optimizer.zero_grad()
        
        # Perform forward pass and compute the loss
        qz, pz, z, xhats, cg_xhats = self.module(x, metadata, expert_id)
        
        loss_dict = self.module.loss(
            x=x, qz=qz, pz=pz, xhats=xhats, cg_xhats=cg_xhats,
            kl_weight=self.kl_annealing_fn.kl_weight, expert_id=expert_id)

        # Perform manual backpropagation
        self.manual_backward(loss_dict[RK.LOSS])

        # Clip gradients for stability
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        
        inactive_modules = []
        if self.module.vae.conditionals:
            for conditional_layer in self.module.vae.conditionals.layers:
                inactive = conditional_layer.unique_conditions - conditional_layer.active_condition_modules
                inactive_modules.extend([conditional_layer.conditions[key] for key in conditional_layer.conditions if key in inactive])
        
        for module in inactive_modules:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # Update the weights
        vae_optimizer.step()
        expert_optimizer.step()
        self.kl_annealing_fn.step()
        
        # Log the loss
        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])
        
    def validation_step(self, batch):
        """
        Validation step for the model.

        Args:
            batch (dict): Batch of data containing inputs and target labels.

        Returns:
            None
        """
        x, metadata, expert_id = batch
        
        qz, pz, z, xhats, cg_xhats = self.module(x, metadata, expert_id, cross_generate=True)
        
        loss_dict = self.module.loss(
            x=x, qz=qz, pz=pz, xhats=xhats, cg_xhats=cg_xhats,
            kl_weight=self.kl_annealing_fn.kl_weight, expert_id=expert_id)
        
        # Log the loss if not in sanity checking phase
        if not self.trainer.sanity_checking:
            self.auto_log(loss_dict, tags=[self.stage_name, expert_id])
        
    # Alias for validation_step method to reuse for testing
    test_step = validation_step
    
    def predict_step(self, batch, batch_idx):
        
        embeddings = self.module.get_latent_embeddings(*batch)
        self.save_predictions(embeddings, batch_idx)
        
    def configure_optimizers(self):
        expert_optimizers = {
            expert_id: torch.optim.Adam(self.module.experts[expert_id].parameters(), lr=1e-3, weight_decay=1e-6)
            for expert_id in self.module.experts
        }
        vae_optimizer =  torch.optim.Adam(self.module.vae.encoder.parameters(), lr=1e-3, weight_decay=1e-6)
        optimizers = [('vae', vae_optimizer)] + [(expert_id, expert_optimizers[expert_id]) for expert_id in expert_optimizers]
        self.optimizer_map = { name: idx for idx, (name, _) in enumerate(optimizers) }
        
        return [optimizer for _, optimizer in optimizers]