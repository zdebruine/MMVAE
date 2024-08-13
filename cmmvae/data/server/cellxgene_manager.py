from typing import Any
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = dict((("train", 0.8), ("val", 0.1), ("test", 0.1)))

OBS_COL_NAMES = (
    "dataset_id",
    "assay", 
    "donor_id",
    "cell_type",
)

OBS_QUERY_VALUE_FILTER = 'is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]'


class CellxgeneManager:
    """
    CellxgeneManager is responsible for managing data loading and processing 
    for the cellxgene dataset. It utilizes tiledbsoma and cellxgene_census 
    libraries to create and handle experiment data pipes.

    Attributes:
        batch_size (int): Size of the data batches.
        seed (int): Random seed for data operations.
        split_weights (dict[str, float]): Weights for splitting data into train, val, and test sets.
        obs_query_value_filter (str): Filter string for querying observational data.
        obs_column_names (tuple[str]): Column names for observational data.
        soma_chunk_size (int): Chunk size for SOMA data.
        census (Any): Reference to the open census data, initialized during setup.
        census_version (str): Version of the cellxgene census data to be used.
        experiment_datapipe (ExperimentDataPipe): The data pipe for handling experiment data.
        datapipes (dict[str, Any]): Data pipes for train, val, and test datasets.
    """

    def __init__(
        self,
        batch_size: int,
        seed: int,
        split_weights=DEFAULT_WEIGHTS,
        obs_query_value_filter: str = OBS_QUERY_VALUE_FILTER,
        obs_column_names: tuple[str] = OBS_COL_NAMES,
        soma_chunk_size: int = None,
        census_version: str = "2023-12-15"
    ):
        """
        Initialize the CellxgeneManager with the specified parameters.

        Args:
            batch_size (int): Size of the data batches.
            seed (int): Random seed for data operations.
            split_weights (dict[str, float], optional): Weights for splitting data into train, val, and test sets. Defaults to DEFAULT_WEIGHTS.
            obs_query_value_filter (str, optional): Filter string for querying observational data. Defaults to OBS_QUERY_VALUE_FILTER.
            obs_column_names (tuple[str], optional): Column names for observational data. Defaults to OBS_COL_NAMES.
            soma_chunk_size (int, optional): Chunk size for SOMA data. Defaults to None.
            census_version (str, optional): Version of the cellxgene census data to be used. Defaults to "2023-12-15".
        """
        self.batch_size = batch_size
        self.obs_query_value_filter = obs_query_value_filter
        self.seed = seed
        self.split_weights = split_weights
        self.obs_column_names = obs_column_names
        self.soma_chunk_size = soma_chunk_size
        self.census = None
        self.census_version = census_version
        
    def setup(self):
        """
        Set up the data manager by opening the cellxgene census and creating experiment data pipes.
        Initializes data pipes for training, validation, and testing based on the split weights.
        """
        import cellxgene_census as cell_census
        import cellxgene_census.experimental.ml as census_ml
        import tiledbsoma as soma

        self.census = cell_census.open_soma(census_version=self.census_version)
        
        self.experiment_datapipe = census_ml.ExperimentDataPipe(
            experiment=self.census["census_data"]["homo_sapiens"],
            measurement_name="RNA",
            X_name="normalized",
            obs_query=soma.AxisQuery(value_filter=self.obs_query_value_filter),
            obs_column_names=self.obs_column_names,
            shuffle=True,
            batch_size=self.batch_size,
            seed=self.seed,
            soma_chunk_size=self.soma_chunk_size,
            use_eager_fetch=False)
        
        if not self.split_weights:
            self.datapipes = {k: self.experiment_datapipe for k in DEFAULT_WEIGHTS.keys()}
        else:
            datapipes = self.experiment_datapipe.random_split(
                seed=self.seed,
                weights=self.split_weights)
            
            self.datapipes = dict((k, v) for k, v in zip(self.split_weights.keys(), datapipes))
    
    def teardown(self):
        """
        Tear down the data manager by closing any open resources held by the cellxgene census.
        """
        if self.census and hasattr(self.census, 'close'):
            self.census.close()
        
    def create_dataloader(self, target: str, num_workers: int):
        """
        Create a data loader for the specified dataset target.

        Args:
            target (str): The target dataset to create a data loader for ('train', 'val', or 'test').
            num_workers (int): Number of workers for data loading.

        Returns:
            Any: A data loader configured for the specified dataset target.

        Raises:
            ValueError: If the target is not in the defined split weights.
        """
        import cellxgene_census.experimental.ml as census_ml

        if target not in self.datapipes:
            raise ValueError(f"target {target} not in {self.split_weights.keys()}")
        dp = self.datapipes[target]
        
        return census_ml.experiment_dataloader(
            dp,
            pin_memory=True,
            num_workers=num_workers,
            # causes OOM error
            # persistent_workers=self.trainer.training and self.hparams.num_workers > 0,
            prefetch_factor=1)
        
    def metadata_to_df(self, metadata: Any) -> pd.DataFrame:
        """
        Convert metadata to a pandas DataFrame using observational encoders.

        Args:
            metadata (Any): Metadata to be converted to DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the decoded metadata.
        """
        obs_encoders = self.experiment_datapipe.obs_encoders
        return pd.DataFrame(
            np.stack([
                obs_encoders[key].inverse_transform(metadata[:, i])
                for i, key in enumerate(obs_encoders, start=1)
            ], axis=1),
            columns=list(obs_encoders.keys())
        )