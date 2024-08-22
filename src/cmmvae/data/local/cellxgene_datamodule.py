from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn

from cmmvae.data.local.cellxgene_datapipe import SpeciesDataPipe
from cmmvae.data.local.multi_modal_loader import MultiModalDataLoader
from cmmvae.data.local.cellxgene_manager import SpeciesManager


class SpeciesDataModule(LightningDataModule):
    """
    A LightningDataModule for handling data loading and preparation for multiple species datasets.

    This module orchestrates the training, validation, testing, and prediction dataloaders
    for multiple species, enabling efficient data management and processing.

    Attributes:
        species (list[Any]): A list of `Any` instances for each species.
        num_workers (int): Number of workers for data loading.
        n_val_workers (int): Number of workers for validation data loading.
        n_test_workers (int): Number of workers for test data loading.
        n_predict_workers (int): Number of workers for prediction data loading.

    """

    def __init__(
        self,
        species: list[SpeciesManager],
        num_workers: int,
        n_val_workers: int = None,
        n_test_workers: int = None,
        n_predict_workers: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.species: list[SpeciesManager] = species
        self.num_workers = num_workers
        self.n_val_workers = n_val_workers if n_val_workers else num_workers
        self.n_test_workers = n_test_workers if n_val_workers else num_workers
        self.n_predict_workers = n_predict_workers if n_val_workers else num_workers

    def setup(self, stage):
        """
        Sets up the data pipelines based on the current stage of the training process.

        Args:
            stage (str): The current stage in the training process, e.g., 'fit', 'validate', 'test', 'predict'.

        """
        if stage in (TrainerFn.FITTING,):
            self._train_datapipe = self.train_datapipe()
            self._val_datapipe = self.val_datapipe()
        elif stage in (TrainerFn.VALIDATING,):
            self._val_datapipe = self.val_datapipe()
        elif stage in (TrainerFn.PREDICTING, TrainerFn.TESTING):
            self._test_datapipe = self.test_datapipe()

    def train_datapipe(self):
        """
        Creates the training data pipeline for each species.

        Returns:
            list: A list of training data pipelines for each species.
        """
        return [species.train_datapipe() for species in self.species]

    def val_datapipe(self):
        """
        Creates the validation data pipeline for each species.

        Returns:
            list: A list of validation data pipelines for each species.
        """
        return [species.val_datapipe() for species in self.species]

    def test_datapipe(self):
        """
        Creates the test data pipeline for each species.

        Returns:
            list: A list of test data pipelines for each species.
        """
        return [species.test_datapipe() for species in self.species]

    @property
    def can_pin_memory(self):
        """
        Determines if memory pinning can be enabled based on the data format.

        Returns:
            bool: True if all species return dense data, allowing for memory pinning.
        """
        return all(species.return_dense for species in self.species)

    def train_dataloader(self):
        """
        Creates the training DataLoader.

        Returns:
            DataLoader: A DataLoader for the training data pipeline.
        """
        dps = list(self.train_datapipe())
        return self.create_dataloader(
            *dps, pin_memory=self.can_pin_memory, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """
        Creates the validation DataLoader.

        Returns:
            DataLoader: A DataLoader for the validation data pipeline.
        """
        dps = list(self.val_datapipe())
        return self.create_dataloader(
            *dps, pin_memory=self.can_pin_memory, num_workers=self.n_val_workers
        )

    def test_dataloader(self):
        """
        Creates the test DataLoader.

        Returns:
            DataLoader: A DataLoader for the test data pipeline.
        """
        dps = list(self.test_datapipe())
        return self.create_dataloader(
            *dps, pin_memory=self.can_pin_memory, num_workers=self.n_test_workers
        )

    def predict_dataloader(self):
        """
        Creates the prediction DataLoader.

        Returns:
            DataLoader: A DataLoader for the prediction data pipeline.
        """
        dps = list(self.test_datapipe())
        return self.create_dataloader(
            *dps, pin_memory=self.can_pin_memory, num_workers=self.n_test_workers
        )

    def create_dataloader(self, *species: SpeciesDataPipe, **kwargs):
        """
        Creates a DataLoader for the given species data pipelines.

        Args:
            species (cmmvae.data.local.SpeciesDataPipe): Data pipelines for the species.
            **kwargs: Additional keyword arguments for the DataLoader.

        Returns:
            DataLoader or MultiModalDataLoader: A DataLoader if a single species pipeline is provided,
            otherwise a MultiModalDataLoader for multiple species.
        """
        dataloaders = [
            DataLoader(
                dataset=dp,
                batch_size=None,
                shuffle=False,
                collate_fn=lambda x: x,
                persistent_workers=False,
                **kwargs
            )
            for dp in species
        ]
        if len(dataloaders) == 1:
            return dataloaders[0]
        return MultiModalDataLoader(*dataloaders)
