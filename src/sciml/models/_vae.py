import torch
import pandas as pd
import numpy as np
import scipy as sp
import pickle
from .base._base_vae_model import BaseVAEModel
from sciml.modules import VAE
from sciml.utils.constants import REGISTRY_KEYS as RK



class VAEModel(BaseVAEModel):
    """
    Variational Autoencoder (VAE) model class.

    Args:
        module (VAE): VAE module to be used in the model.
        **kwargs: Additional keyword arguments for the base VAE model.
    """
    
    def __init__(self, module: VAE, **kwargs):
        super().__init__(module, **kwargs)
    
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
        x = batch[RK.X]
        metadata = batch[RK.METADATA]
        _, z = self.module.encode(x)
        z_star = self.module.after_reparameterize(z, metadata)
        
        predictions = {
            RK.Z: z,
            RK.METADATA: metadata, 
        }
        
        if z_star == z:
            return predictions
        
        predictions.update({ RK.Z_STAR: z_star })
        return predictions
        
    def save_predictions(self, predictions):  
        
        stacked_predictions = {}
        for key in predictions[0].keys():
            if isinstance(predictions[0][key], pd.DataFrame):
                stacked_predictions[key] = pd.concat([prediction[key] for prediction in predictions])
            else:
                stacked_predictions[key] = torch.cat([prediction[key].numpy() for prediction in predictions], dim=0)
        
        for key in stacked_predictions:
            if key == RK.METADATA:
                continue
            self.save_latent_predictions(
                embeddings=stacked_predictions[key], 
                metadata=stacked_predictions[RK.METADATA], 
                embeddings_name=f"{key}_embeddings.npz",
                metadata_name=f"{key}_metadata.pkl")
