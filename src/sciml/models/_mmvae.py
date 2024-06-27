from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from .base import utils
from sciml.utils.constants import REGISTRY_KEYS as RK
from .base import BaseVAEModel
from sciml.modules import MMVAE

class MMVAEModel(BaseVAEModel):
    """
    Multi-Modal Variational Autoencoder (MMVAE) model for handling expert-specific data.

    Args:
        module (MMVAE): Multi-Modal VAE module.
        **kwargs: Additional keyword arguments for the base VAE model.

    Attributes:
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
    """
    
    def __init__(self, module: MMVAE, **kwargs):
        super().__init__(module, **kwargs)
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
        # Retrieve optimizers
        shared_opt, human_opt, mouse_opt = self.optimizers()
        
        # Select expert-specific optimizer based on the expert ID in the batch
        expert_opt = human_opt if batch[RK.EXPERT_ID] == RK.HUMAN else mouse_opt

        # Zero the gradients for the shared and expert-specific optimizers
        shared_opt.zero_grad()
        expert_opt.zero_grad()
    
        # Perform forward pass and compute the loss
        model_inputs, model_outputs, loss = self(batch, model_input_kwargs={'target': batch[RK.EXPERT_ID]}) 
        
        # Perform manual backpropagation
        self.manual_backward(loss[RK.LOSS])
        
        # Clip gradients for stability
        self.clip_gradients(shared_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        
        # Update the weights
        shared_opt.step()
        expert_opt.step()
        
        # Log the loss
        self.auto_log(loss, tags=[str(self.trainer.state.stage), model_inputs[RK.EXPERT]])

    def on_train_epoch_end(self) -> None:
        """
        Actions to perform at the end of each training epoch.

        Returns:
            None
        """
        # Reset the training dataloader
        self.trainer.train_dataloader.reset()
        
    def validation_step(self, batch):
        """
        Validation step for the model.

        Args:
            batch (dict): Batch of data containing inputs and target labels.

        Returns:
            None
        """
        # Perform forward pass and compute the loss with cross-generation loss
        model_inputs, _, loss = self(batch, loss_kwargs={'use_cross_gen_loss': True})
        
        # Log the loss if not in sanity checking phase
        if not self.trainer.sanity_checking:
            self.auto_log(loss, tags=[str(self.trainer.state.stage), model_inputs[RK.EXPERT]])
    
    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of each validation epoch.

        Returns:
            None
        """
        # Reset the validation dataloader
        self.trainer.val_dataloaders.reset()
        
    # Alias for validation_step method to reuse for testing
    test_step = validation_step
