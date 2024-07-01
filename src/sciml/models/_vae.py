import torch
import pandas as pd
import numpy as np
import pickle
from .base._base_vae_model import BaseVAEModel
from sciml.modules import SimpleVAE
from sciml.utils.constants import REGISTRY_KEYS as RK

class VAEModel(BaseVAEModel):
    """
    Variational Autoencoder (VAE) model class.

    Args:
        module (SimpleVAE): SimpleVAE module to be used in the model.
        **kwargs: Additional keyword arguments for the base VAE model.
    """
    
    def __init__(self, module: SimpleVAE, **kwargs):
        super().__init__(module, **kwargs)
        
        self.predict_container_z = []
        self.predict_container_metadata = []
    @property
    def example_input_array(self):
        """
        Provides an example input array for tensorboard log_graph.

        Returns:
            dict: Example input data.
        """
        # Wrapped in a dictionary because tensorboard unpacks when passing to forward pass when it sees a tuple
        # Made Y and metadata tensors because jit.trace cannot track null values 
        return { 'batch': { RK.X: torch.rand((self.hparams.batch_size, 60664)) }, 'compute_loss': False }
    
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
    
    def on_predict_epoch_end(self):
        
        npz = torch.cat(self.predict_container_z).numpy()
        metadata = pd.concat(metadata, axis=0)
        
        np.save(f"{self.logger.log_dir}/z_values.npz", npz)
        metadata.to_pickle(f"{self.logger.log_dir}/metadata.pkl")
        
    def predict_step(self, batch, batch_idx):
        x = batch[RK.X]
        metadata = batch[RK.METADATA]
        z = self.module.encode(x)
        self.predict_container_z.append(z)
        self.predict_container_metadata.append(metadata)