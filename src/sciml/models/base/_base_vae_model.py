from typing import Iterable, Literal, Optional, Union, Any, Callable
import torch
import lightning.pytorch as pl
from . import utils

from sciml.utils.constants import REGISTRY_KEYS as RK

from sciml.modules.base import BaseModule

class BaseVAEModel(pl.LightningModule):
    """
    Base class for Variational Autoencoder (VAE) models, extending PyTorch Lightning's LightningModule.

    This class provides a structured approach to experiment tracking, auto-optimization,
    and modularized forward pass handling for VAE architectures. It serves as a foundation
    for building VAE models and their derivatives, reducing redundancy in trace code and logging.

    Args:
        module (BaseModule): The module to be used in the forward pass.
        batch_size (int, optional): Batch size for logging purposes only. Defaults to 128.
        record_embeddings (bool, optional): Whether to record embeddings (z and z*star). Defaults to False.
        record_gradients (bool, optional): Whether to record gradients of the model. Defaults to False.
        save_embeddings_interval (int, optional): Interval to save embeddings to avoid clogging TensorBoard. Defaults to 25.
        configure_optimizer_kwargs (dict, optional): Keyword arguments to be passed to configure_optimizers. Defaults to an empty dict.
        gradient_record_cap (int, optional): Cap on the number of gradients to record to prevent clogging TensorBoard. Defaults to 20.

    Attributes:
        get_module_input_kwargs (Callable): Returns input arguments for the forward pass based on the data registry and additional kwargs.
        example_input_array (tuple): A tuple of positional arguments to be passed to the forward pass.
        get_adata_latent_representations (Callable): Takes AnnData and returns latent representations.
    """

    def __init__(
        self,
        module: BaseModule,
        batch_size: int = 128,
        record_embeddings: bool = False,
        record_gradients: bool = False,
        save_embeddings_interval: int = 25,
        configure_optimizer_kwargs: dict[str, Any] = {},
        gradient_record_cap: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['module'], logger=False)
        
        self.module = module
        self.batch_size = batch_size
        self._configure_optimizer_kwargs = configure_optimizer_kwargs
        self.save_embeddings_interval = save_embeddings_interval
        self.record_embeddings = record_embeddings
        self.record_gradients = record_gradients
        self.gradient_record_cap = gradient_record_cap
        
    @property
    def stage_name(self):
        """
        Returns the current stage name of the trainer.

        Returns:
            str: The stage name (training, validation, sanity_checking, prediction, test).
        """
        if self.trainer.training:
            return 'training'
        elif self.trainer.validating:
            return 'validation'
        elif self.trainer.sanity_checking:
            return 'sanity_checking'
        elif self.trainer.predicting:
            return 'prediction'
        elif self.trainer.testing:
            return 'test'
    
    def save_embeddings(self, model_inputs, model_outputs):
        """
        Abstract method to save embeddings from the model during the forward pass.
        """
        pass
        
    def save_gradient(self, tag, grad):
        """
        Save a gradient to TensorBoard.

        Args:
            tag (str): Tag for the gradient.
            grad (torch.Tensor): The gradient tensor.
        """
        self.logger.experiment.add_histogram(
            tag=tag, values=grad, global_step=self.trainer.global_step
        )
        
    def save_gradients(self):
        """
        Save the gradients of the model to TensorBoard.

        This is a no-op if the number of named parameters exceeds gradient_record_cap to prevent clogging TensorBoard.
        """
        if len(self.named_parameters()) > self.gradient_record_cap:
            self.record_gradients = False
            return 
        for k, v in self.named_parameters():
            if v.requires_grad:
                self.save_gradient(k, v.grad)
    
    @property
    def record_embeddings(self):
        """
        Determine whether to record embeddings during the forward pass.

        Returns:
            bool: True if embeddings should be recorded, False otherwise.
        """
        return self._record_embeddings and not self.trainer.sanity_checking
        
    @record_embeddings.setter
    def record_embeddings(self, value: bool):
        self._record_embeddings = value
        
    @property
    def record_gradients(self):
        """
        Determine whether to record gradients during the forward pass.

        Returns:
            bool: True if gradients should be recorded, False otherwise.
        """
        return self._record_gradients and not self.trainer.evaluating
    
    @record_gradients.setter
    def record_gradients(self, value: bool):
        self._record_gradients = value

    def on_before_optimizer_step(self, optimizer):
        """
        Save gradients before the optimizer step if necessary.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer being used.
        """
        if self.record_gradients and self.trainer.global_step % self.save_embeddings_interval == 0:
            self.save_embeddings()
    
    def forward(
        self, 
        batch: Any, 
        compute_loss: bool = True,
        record_embeddings: Optional[bool] = None,
        record_gradients: Optional[bool] = None,
        module_input_kwargs = {}, 
        loss_kwargs = {}
    ):
        """
        Forward method for the model.

        This method processes the input batch and passes it through the module,
        optionally computing loss and recording embeddings or gradients.

        Args:
            batch (Any): The input batch.
            compute_loss (bool, optional): Whether to compute and return the loss. Defaults to True.
            record_embeddings (Optional[bool], optional): Override for recording embeddings. Defaults to None.
            record_gradients (Optional[bool], optional): Override for recording gradients. Defaults to None.
            module_input_kwargs (dict, optional): Additional keyword arguments for the module's input. Defaults to an empty dict.
            loss_kwargs (dict, optional): Additional keyword arguments for the loss computation. Defaults to an empty dict.

        Returns:
            tuple: The model inputs, model outputs, and loss (if computed).
        """
        # Allow for individualized control on forward pass
        if record_embeddings is not None:  # optional override to save embeddings
            self.record_embeddings = record_embeddings
        if record_gradients is not None:  # optional override to record gradients
            self.record_gradients = record_gradients
        
        # Get module inputs for the forward pass in the form of args and kwargs
        model_inputs = self.module.get_module_inputs(batch, **module_input_kwargs)
        args, kwargs = model_inputs  # Split model inputs into args and kwargs
        model_outputs = self.module(*args, **kwargs)  # Pass args and kwargs to the forward method of the module
        
        # Record embeddings of the forward pass if necessary
        if self.record_embeddings:
            self.save_embeddings(model_inputs, model_outputs)
        
        # Compute the loss of the module if necessary
        loss = None
        if compute_loss:
            loss = self.module.loss(model_inputs, model_outputs, **loss_kwargs)
        
        return model_inputs, model_outputs, loss
    
    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        Returns:
            list: A list of optimizers.
        """
        return self.module.configure_optimizers(**self._configure_optimizer_kwargs)
        
    def get_adata_latent_representations(
        self,
        adata,
        batch_size
    ):
        """
        Get latent representations from AnnData.

        Args:
            adata (AnnData): Annotated data matrix.
            batch_size (int): Batch size for data loading.

        Returns:
            np.ndarray: Latent representations.
        """
        from sciml.data.server import AnnDataDataset, collate_fn
        from torch.utils.data import DataLoader
        from lightning.pytorch.trainer import Trainer

        dataset = AnnDataDataset(adata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        zs = []
        self.eval()
        with torch.no_grad():
            for batch_dict in dataloader:
                for key in batch_dict.keys():
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].to('cuda')
                    
                predict_outputs = self.predict_step(batch_dict, None)
                zs.append(predict_outputs[RK.Z])
        
        return torch.cat(zs).numpy()
    
    def auto_log(    
        self,
        log_dict: dict[str, torch.Tensor],
        tags: Iterable[str] = [],
        sep: str = "/",
        key_pos: Literal['first', 'last'] = 'first',
        log_sanity_checking: bool = False,
    ):
        """
        Automatically log a dictionary of metrics to TensorBoard.

        By default, this avoids logging during sanity checking unless overridden by setting log_sanity_checking to True.

        Args:
            log_dict (dict[str, torch.Tensor]): Dictionary of metrics to log.
            tags (Iterable[str], optional): Iterable of strings to join as a separator. Defaults to an empty list.
            sep (str, optional): Separator string for tags. Defaults to "/".
            key_pos (Literal['first', 'last'], optional): Position to place the original key of the log dictionary. Defaults to 'first'.
            log_sanity_checking (bool, optional): Whether to log during sanity checking. Defaults to False.
        """
        # Avoid logging during sanity checking unless necessary
        if self.trainer.sanity_checking and not log_sanity_checking:
            return
        
        # Tag log dict to differentiate losses
        log_dict = utils.tag_log_dict(log_dict, tags, sep, key_pos)
        
        self.log_dict(
            log_dict, 
            on_step=self.trainer.training, 
            on_epoch=True, 
            logger=True, 
            batch_size=self.batch_size
        )
