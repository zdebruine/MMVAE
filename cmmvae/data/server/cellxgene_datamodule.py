from typing import Any, Sequence
import lightning as L

from .cellxgene_manager import (
    CellxgeneManager, 
    OBS_COL_NAMES, 
    OBS_QUERY_VALUE_FILTER, 
    DEFAULT_WEIGHTS
)

from cmmvae.constants import REGISTRY_KEYS as RK

class CellxgeneDataModule(L.LightningDataModule):
    """
    CellxgeneDataModule is a PyTorch Lightning data module designed to manage data loading 
    for cellxgene datasets. It provides training, validation, test, and prediction data loaders, 
    leveraging the CellxgeneManager for handling dataset configurations and loading.

    Attributes:
        cellx_manager (CellxgeneManager): An instance of CellxgeneManager for managing data operations.
    """

    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        obs_query_value_filter: str = OBS_QUERY_VALUE_FILTER,
        obs_column_names: Sequence[str] = OBS_COL_NAMES,
        split_weights: dict[str, float] = DEFAULT_WEIGHTS,
        soma_chunk_size: int = None,
        num_workers: int = 3
    ):
        """
        Initialize the CellxgeneDataModule with dataset parameters.

        Args:
            batch_size (int, optional): Size of data batches. Defaults to 128.
            seed (int, optional): Random seed for data splitting and shuffling. Defaults to 42.
            obs_query_value_filter (str, optional): Filter string for querying observational data. Defaults to OBS_QUERY_VALUE_FILTER.
            obs_column_names (Sequence[str], optional): Column names for observational data. Defaults to OBS_COL_NAMES.
            split_weights (dict[str, float], optional): Weights for train, validation, and test data splits. Defaults to DEFAULT_WEIGHTS.
            soma_chunk_size (int, optional): Chunk size for SOMA data. Defaults to None.
            num_workers (int, optional): Number of workers for data loading. Defaults to 3.
        """
        super(CellxgeneDataModule, self).__init__()
        self.cellx_manager = CellxgeneManager(
            batch_size, seed, split_weights, obs_query_value_filter,
            obs_column_names, soma_chunk_size
        )
        self.save_hyperparameters(logger=True)
        
    def setup(self, stage: str):
        """
        Set up the data module by initializing the CellxgeneManager.

        Args:
            stage (str): The stage of training (e.g., 'fit', 'test', 'predict').
        """
        self.cellx_manager.setup()
        
    def teardown(self, stage: str):
        """
        Tear down the data module by closing any resources held by the CellxgeneManager.

        Args:
            stage (str): The stage of training (e.g., 'fit', 'test', 'predict').
        """
        self.cellx_manager.teardown()
        
    def train_dataloader(self) -> Any:
        """
        Create a data loader for training data.

        Returns:
            Any: A data loader for training data, configured with the specified number of workers.
        """
        return self.cellx_manager.create_dataloader('train', self.hparams.num_workers)
    
    def val_dataloader(self) -> Any:
        """
        Create a data loader for validation data.

        Returns:
            Any: A data loader for validation data, using 2 workers.
        """
        return self.cellx_manager.create_dataloader('val', 2)
        
    def test_dataloader(self) -> Any:
        """
        Create a data loader for test data.

        Returns:
            Any: A data loader for test data, using 2 workers.
        """
        return self.cellx_manager.create_dataloader('test', 2)
        
    def predict_dataloader(self) -> Any:
        """
        Create a data loader for prediction.

        Returns:
            Any: A data loader for prediction data, using the test set configuration.
        """
        return self.cellx_manager.create_dataloader('test', 2)
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        Hook to modify batches before they are transferred to the GPU. Adds metadata to each batch.

        Args:
            batch (Any): The input batch data consisting of features and labels.
            dataloader_idx (int): The index of the dataloader calling this hook.

        Returns:
            Any: A tuple containing the input batch features, metadata, and the species label.
        """
        x_batch, labels = batch
        
        metadata = self.cellx_manager.metadata_to_df(labels)
        
        return x_batch, metadata, 'human'