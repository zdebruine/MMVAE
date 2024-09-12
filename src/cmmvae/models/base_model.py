from typing import Iterable, Literal, Optional, Union
import os

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.modules.base import KLAnnealingFn
import cmmvae.modules.base.init as init


def tag_log_dict(
    log_dict: dict[str, torch.Tensor],
    tags: Iterable[str] = [],
    sep: str = "/",
    key_pos: Union[Literal["first"], Literal["last"]] = "first",
) -> dict[str, torch.Tensor]:
    """
    Annotates loss output keys with specified tags.

    Args:
        log_dict (Dict[str, torch.Tensor]): A dictionary containing loss outputs.
        tags (str): Tags to append or prepend to the keys.
        sep (str): Separator used to concatenate tags and keys.
        key_first (bool): If True, places the key before the tags. If False, places the tags before the key.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with updated keys based on the tags and separator.
    """

    if key_pos not in ("last", "first"):
        raise ValueError("Key position {key_pos} is not supported!")

    tags_str = sep.join(tags)

    def key_generator(key):
        if not tags_str:
            return key
        elif key_pos == "first":
            return f"{key}{sep}{tags_str}"
        elif key_pos == "last":
            return f"{tags_str}{sep}{key}"
        else:
            raise RuntimeError("Unexpected code block executed!")

    return {key_generator(key): value for key, value in log_dict.items()}


class BaseModel(pl.LightningModule):
    """
    This class serves as a wrapper to `LightningModule` providing methods for experiment tracking and training.


    Attributes:
        example_input_array (tuple): A tuple of positional arguments to be passed to the forward pass for graph creation.
    """

    def __init__(
        self,
        batch_size: int = 128,
        record_gradients: bool = False,
        save_gradients_interval: int = 25,
        gradient_record_cap: int = 20,
        kl_annealing_fn: Optional[KLAnnealingFn] = None,
        predict_dir: str = "",
        predict_save_interval: int = 600,
        initial_save_index: int = -1,
        use_he_init_weights: bool = True,
    ):
        """
        Initializes a BaseModel instance registering KLAnnealingFn and initializing weights.

        .. warning::
            Inorder for module parameters to be picked up subclasses must call :attr:`init_weights` to initialize weights.

        Args:
            batch_size (int, optional): Batch size for logging purposes only. Defaults to 128.
            record_gradients (bool, optional): Whether to record gradients of the model. Defaults to False.
            save_gradients_interval (int): Interval of steps to save gradients. Defaults to 25.
            gradient_record_cap (int, optional): Cap on the number of gradients to record to prevent clogging TensorBoard. Defaults to 20.
            kl_annealing_fn (KLAnnealingFn, optional): Annealing function used for kl_weight. Defaults to `KLAnnealingFn(1.0)`
            predict_dir (str): Directory to save predictions. If not absolute path then saved within Tensorboard log_dir. Defaults to "".
            predict_save_interval (int): Interval to save embeddings and metadata to prevent OOM Error. Defaults to 600.
            initial_save_index (int): The starting point for predictions index when saving (ie z_embeddings_0.npz for -1). Defaults to -1.
            use_he_init_weights (bool): Initialize weights using He initialization. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["module"], logger=False)

        self.batch_size = batch_size
        self.record_gradients: bool = record_gradients
        self.save_gradients_interval: int = save_gradients_interval
        self.gradient_record_cap: int = gradient_record_cap
        self.predict_dir = predict_dir
        self.predict_save_interval = predict_save_interval

        self._running_predictions = []
        self._curr_save_idx = initial_save_index

        if not kl_annealing_fn:
            kl_annealing_fn = KLAnnealingFn(1.0)

        self.kl_annealing_fn = kl_annealing_fn
        self._use_he_init_weights = use_he_init_weights

    def init_weights(self):
        """Initialize model parameters with He initialization."""
        if self._use_he_init_weights:
            init.he_init_weights(self)

    def save_latent_predictions(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        embeddings_path: str = "embeddings.npz",
        metadata_path: str = "metadata.pkl",
    ):
        """
        Saves embeddings and metadata to disk.

        .. note::
            if path supplied in :attr:`embeddings_path` or :attr:`metadata_path`
            is not an absolute path then the current logger.log_dir is used.

        Args:
            embeddings (np.ndarray): Model embeddings
            metadata (pd.DataFrame): Associate metadata
            embeddings_path (str): Name of embeddings file
            metadata_path (str): name of metadata file
        """
        for path in (metadata_path, embeddings_path):
            if not os.path.isabs(path):
                path = os.path.join(self.logger.log_dir, path)
            directory = os.path.dirname(path)

            os.makedirs(directory, exist_ok=True)

            if path.endswith(".npz"):
                np.savez(path, embeddings=embeddings)
            if path.endswith(".pkl"):
                metadata.to_pickle(path)

    @property
    def stage_name(self):
        """
        Returns the current stage name of the trainer.

        Returns:
            str: The stage name (training, validation, sanity_checking, prediction, test).
        """
        if self.trainer.training:
            return "training"
        elif self.trainer.validating:
            return "validation"
        elif self.trainer.sanity_checking:
            return "sanity_checking"
        elif self.trainer.predicting:
            return "prediction"
        elif self.trainer.testing:
            return "test"

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
    def record_gradients(self):
        """
        Determine whether to record gradients during the forward pass.

        Returns:
            bool: True if gradients should be recorded, False otherwise.
        """
        return self._record_gradients and self.trainer and not self.trainer.evaluating

    @record_gradients.setter
    def record_gradients(self, value: bool):
        self._record_gradients = value

    def on_before_optimizer_step(self, optimizer):
        """
        Save gradients before the optimizer step if necessary.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer being used.
        """
        if (
            self.record_gradients
            and self.trainer
            and self.trainer.global_step % self.save_gradients_interval == 0
        ):
            self.save_gradients()

    def get_adata_latent_representations(self, adata, batch_size):
        """
        Get latent representations from AnnData.

        Args:
            adata (Any): Annotated data matrix.
            batch_size (int): Batch size for data loading.

        Returns:
            np.ndarray: Latent representations.
        """
        from cmmvae.data.server import AnnDataDataset, collate_fn
        from torch.utils.data import DataLoader

        dataset = AnnDataDataset(adata)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        zs = []
        self.eval()
        with torch.no_grad():
            for batch_dict in dataloader:
                for key in batch_dict.keys():
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].to("cuda")

                predict_outputs = self.predict_step(batch_dict, None)
                zs.append(predict_outputs[RK.Z])

        return torch.cat(zs).numpy()

    def auto_log(
        self,
        log_dict: dict[str, torch.Tensor],
        tags: Iterable[str] = [],
        sep: str = "/",
        key_pos: Literal["first", "last"] = "first",
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
        if self.trainer and self.trainer.sanity_checking and not log_sanity_checking:
            return

        # Tag log dict to differentiate losses
        log_dict = tag_log_dict(log_dict, tags, sep, key_pos)

        self.log_dict(
            log_dict,
            on_step=self.trainer.training,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )

    def on_predict_epoch_start(self):
        """Clear predictions on epoch start"""
        self._running_predictions.clear()

    def on_predict_epoch_end(self):
        """Save and clear leftover predictions after epoch"""
        if self._running_predictions:
            self._save_paired_predictions()
        self._running_predictions.clear()

    def save_predictions(self, predictions: np.ndarray, idx: int):
        """
        Accumulate predictions and save periodically dicated by :attr:`predict_save_interval`.

        Args:
            predictions (np.ndarray): Predictions to be saved.
            idx (int): Index of save prediction (To keep track of save interval).
        """
        self._running_predictions.append(predictions)

        div, mod = divmod(idx + 1, self.predict_save_interval)
        if mod == 0:
            self._save_paired_predictions()

    def _save_paired_predictions(self):
        self._curr_save_idx += 1
        stacked_predictions = {}

        for key in self._running_predictions[0].keys():
            if isinstance(self._running_predictions[0][key], pd.DataFrame):
                stacked_predictions[key] = pd.concat(
                    [prediction[key] for prediction in self._running_predictions]
                )
            else:
                stacked_predictions[key] = (
                    torch.cat(
                        [prediction[key] for prediction in self._running_predictions],
                        dim=0,
                    )
                    .cpu()
                    .numpy()
                )

        for key in stacked_predictions:
            if RK.METADATA in key:
                continue

            self.save_latent_predictions(
                embeddings=stacked_predictions[key],
                metadata=stacked_predictions[f"{key}_{RK.METADATA}"],
                embeddings_path=os.path.join(
                    self.predict_dir, f"{key}_embeddings_{self._curr_save_idx}.npz"
                ),
                metadata_path=os.path.join(
                    self.predict_dir, f"{key}_metadata_{self._curr_save_idx}.pkl"
                ),
            )
        self._running_predictions.clear()
