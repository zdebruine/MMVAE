from ._base_model import BaseModel
from sciml.modules import CMMVAE
from sciml.constants import REGISTRY_KEYS as RK



class CMMVAEModel(BaseModel):
    """
    Multi-Modal Variational Autoencoder (MMVAE) model for handling expert-specific data.

    Args:
        module (MMVAE): Multi-Modal VAE module.
        **kwargs: Additional keyword arguments for the base VAE model.

    Attributes:
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
    """
    
    def __init__(self, cmmvae: CMMVAE, **kwargs):
        super().__init__(**kwargs)
        self.cmmvae = cmmvae
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
        shared_opt, human_opt, mouse_opt = self.optimizers()

        # Select expert-specific optimizer based on the expert ID in the batch
        expert_opt = human_opt if expert_id == RK.HUMAN else mouse_opt
        
        # Zero the gradients for the shared and expert-specific optimizers
        shared_opt.zero_grad()
        expert_opt.zero_grad()
        
        # Perform forward pass and compute the loss
        qz, pz, z, x_hats = self.cmmvae(x, metadata, expert_id)
        
        loss_dict = self.cmmvae.loss(x, expert_id, qz, pz, x_hats, self.kl_annealing_fn.kl_weight)

        # Perform manual backpropagation
        self.manual_backward(loss_dict[RK.LOSS])

        # Clip gradients for stability
        self.clip_gradients(shared_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        
        # Update the weights
        shared_opt.step()
        expert_opt.step()
        
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
        
        qz, pz, z, x_hats = self.cmmvae(x, metadata, expert_id)
        
        loss_dict = self.cmmvae.loss(x, expert_id, qz, pz, x_hats, self.kl_annealing_fn.kl_weight, compute_cross_gen_loss=True)
        
        # Log the loss if not in sanity checking phase
        if not self.trainer.sanity_checking:
            self.auto_log(loss_dict, tags=[self.stage_name, batch[RK.EXPERT_ID]])
        
    # Alias for validation_step method to reuse for testing
    test_step = validation_step
    
    def predict_step(self, batch, batch_idx):
        
        x, metadata, expert_id = batch
        embeddings = self.cmmvae.get_latent_embeddings()
        self.save_predictions(embeddings)