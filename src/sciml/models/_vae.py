from typing import Any
import torch
import pandas as pd
import numpy as np
import scipy as sp
import pickle
from .base._base_vae_model import BaseVAEModel
from sciml.modules import VAE
from sciml.utils.constants import REGISTRY_KEYS as RK

# prediction_kwargs = { 'conditions': { 'assay': "10x 3\' v3", } }

class VAEModel(BaseVAEModel):
    """
    Variational Autoencoder (VAE) model class.

    Args:
        module (VAE): VAE module to be used in the model.
        **kwargs: Additional keyword arguments for the base VAE model.
    """

    def __init__(self, module: VAE, save_interval: int = 500, **kwargs):
        super().__init__(module, **kwargs)
        self.save_interval = save_interval
        self.predictions = []
        self.predictions_saved = False

        
    def step(self, batch, batch_idx):
        """
        Common step function for training, validation, and testing.

        Args:
            batch (dict): Batch of data containing inputs and target labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Evidence Lower Bound (ELBO) loss.
        """
        
        # Perform forward pass and compute the loss
        _, _, loss = self(batch, compute_loss=True, loss_kwargs={'kl_weight': self.kl_annealing_fn.kl_weight})
        
        # Extract ELBO loss
        elbo = loss[RK.LOSS]
        
        # Log the loss
        self.auto_log(loss, tags=[self.stage_name, batch[RK.EXPERT_ID]])
        
        if self.trainer.training:
            self.kl_annealing_fn.step()
        
        return elbo
    
    # Alias step method for training, validation, and testing steps
    training_step = step
    validation_step = step
    test_step = step
        
    def predict_step(self, batch, batch_idx):
        latent_embeddings = self.module.get_latent_embeddings(batch[RK.X], batch[RK.METADATA], **self.prediction_kwargs)
        self.predictions.append(latent_embeddings)
        
        div, mod = divmod(batch_idx, self.save_interval)
        if mod == 0:
            self.last_div = div
            self.save_predictions(self.predictions, div)
            self.predictions.clear()
    
    def on_predict_epoch_start(self):
        self.predictions.clear()
        
    def on_predict_epoch_end(self):
        if self.predictions:
            self.save_predictions(self.predictions, self.last_div + 1)
        self.predictions.clear()
        self.predictions_saved = True
        
    def save_predictions(self, predictions, idx: Any = None):  
        if self.predictions_saved:
            import warnings
            warnings.warn("save_predictions is a no-op predictions already saved")
            return
        stacked_predictions = {}
        for key in predictions[0].keys():
            if isinstance(predictions[0][key], pd.DataFrame):
                stacked_predictions[key] = pd.concat([prediction[key] for prediction in predictions])
            else:
                stacked_predictions[key] = torch.cat([prediction[key] for prediction in predictions], dim=0).cpu().numpy()

        for key in stacked_predictions:
            if RK.METADATA in key:
                continue
            
            self.save_latent_predictions(
                embeddings=stacked_predictions[key], 
                metadata=stacked_predictions[f"{key}_{RK.METADATA}"], 
                embeddings_path=f"samples/{key}_embeddings_{idx}.npz",
                metadata_path=f"samples/{key}_metadata_{idx}.pkl")
