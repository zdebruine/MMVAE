from typing import Callable, Literal, Optional, Union
import random
import pickle

import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch
from torchdata.datapipes.iter import FileLister, IterDataPipe, Zipper, Multiplexer
from torch.utils.data import functional_datapipe


class safe_functional_datapipe(functional_datapipe):
    """
    Wraps functional_datapipe registration in try/except.
    When module gets reloaded the datapipes are reregistered and this causes
    pytorch to throw "Unable to add DataPipe function name
    {function_name} as it is already taken" exception.
    """

    def __call__(self, cls):
        res = None
        try:
            res = super().__call__(cls)
        except Exception as e:
            if "already taken" not in str(e):
                raise e
        return res


@safe_functional_datapipe("load_matrix_and_dataframe")
class LoadIndexMatchedCSRMatrixAndDataFrameDataPipe(IterDataPipe):
    """
    A DataPipe for loading a CSR matrix and its corresponding DataFrame from file paths.

    This DataPipe takes a source DataPipe that yields tuples of file paths, loads the
    sparse matrix and metadata, and yields them as tuples.

    Attributes:
        source_dp (IterDataPipe): The source DataPipe providing file paths.
        verbose (bool): If True, prints the file paths being loaded.
    """

    def __init__(self, source_datapipe, verbose: bool = False):
        """
        Initializes the DataPipe with a source DataPipe and verbosity setting.

        Args:
            source_datapipe (IterDataPipe): The source DataPipe yielding file paths.
            verbose (bool): If True, enables verbose output. Defaults to False.
        """
        super().__init__()
        self.source_dp = source_datapipe
        self.verbose = verbose

    def __iter__(self):
        """
        Iterates over the source DataPipe, loading and yielding the CSR matrix and DataFrame.

        Yields:
            tuple: A tuple containing a scipy sparse matrix and a metadata DataFrame.

        Raises:
            Exception: If there is an error loading the files.
        """
        for npz_path, metadata_path in self.source_dp:
            if self.verbose:
                print(f"Loading file path: {npz_path}, {metadata_path}", flush=True)

            sparse_matrix = None
            metadata = None

            try:
                with open(npz_path, "rb") as npz_file:
                    sparse_matrix = sp.load_npz(npz_file)

                with open(metadata_path, "rb") as metadata_file:
                    if ".pkl" in metadata_path:
                        metadata = pickle.load(metadata_file)
            except Exception as e:
                print(f"Error loading files: {e}")
                raise

            yield (sparse_matrix, metadata)


@safe_functional_datapipe("shuffle_matrix_and_dataframe")
class ShuffleCSRMatrixAndDataFrameDataPipe(IterDataPipe):
    """
    A DataPipe for shuffling rows of a CSR matrix and a corresponding DataFrame.

    This DataPipe takes a source DataPipe that yields tuples of a CSR matrix and DataFrame,
    applies a random permutation to both, and yields the shuffled results.

    Attributes:
        source_dp (IterDataPipe): The source DataPipe providing the CSR matrix and DataFrame.
    """

    def __init__(self, source_datapipe):
        """
        Initializes the DataPipe with a source DataPipe.

        Args:
            source_datapipe (IterDataPipe): The source DataPipe yielding a CSR matrix and DataFrame.
        """
        super().__init__()
        self.source_dp = source_datapipe

    def __iter__(self):
        """
        Iterates over the source DataPipe, shuffling and yielding the CSR matrix and DataFrame.

        Yields:
            tuple: A tuple containing a shuffled scipy sparse matrix and DataFrame.
        """
        for sparse_matrix, dataframe in self.source_dp:
            permutation = np.random.permutation(sparse_matrix.shape[0])

            dataframe = dataframe.iloc[permutation].reset_index(drop=True)
            sparse_matrix = sparse_matrix[permutation]

            yield (sparse_matrix, dataframe)


@safe_functional_datapipe("batch_csr_matrix_and_dataframe")
class SparseCSRMatrixBatcherDataPipe(IterDataPipe):
    """
    A DataPipe for batching a CSR matrix and corresponding DataFrame.

    This DataPipe creates batches of a specified size from a CSR matrix and DataFrame,
    yielding them as torch sparse tensors.

    Args:
        batch_size (int): The size of each batch.
        allow_partials (bool): Whether to allow partial batches.
        return_dense (bool): Whether to return dense tensors.
    """

    def __init__(
        self,
        source_datapipe,
        batch_size: int,
        allow_partials: bool = False,
        return_dense: bool = False,
    ):
        """
        Initializes the DataPipe with a source DataPipe and batch settings.

        Args:
            source_datapipe (IterDataPipe): The source DataPipe yielding CSR matrix and DataFrame.
            batch_size (int): The size of each batch.
            allow_partials (bool): Whether to allow partial batches. Defaults to False.
            return_dense (bool): Whether to return dense tensors. Defaults to False.
        """
        super(SparseCSRMatrixBatcherDataPipe, self).__init__()

        self.source_datapipe = source_datapipe
        self.batch_size = batch_size
        self.allow_partials = allow_partials
        self.return_dense = return_dense

    def __iter__(self):
        """
        Iterates over the source DataPipe, creating and yielding batches of tensors and metadata.

        Yields:
            tuple: A tuple containing a torch sparse tensor and metadata DataFrame.
        """
        for sparse_matrix, dataframe in self.source_datapipe:
            n_samples = sparse_matrix.shape[0]

            for i in range(0, n_samples, self.batch_size):
                data_batch = sparse_matrix[i : i + self.batch_size]

                if data_batch.shape[0] != self.batch_size and not self.allow_partials:
                    continue

                tensor = torch.sparse_csr_tensor(
                    crow_indices=data_batch.indptr,
                    col_indices=data_batch.indices,
                    values=data_batch.data,
                    size=data_batch.shape,
                )

                if isinstance(dataframe, pd.DataFrame):
                    metadata = dataframe.iloc[i : i + self.batch_size].reset_index(
                        drop=True
                    )

                if self.return_dense:
                    tensor = tensor.to_dense()

                yield tensor, metadata


@safe_functional_datapipe("transform")
class TransformDataPipe(IterDataPipe):
    """
    A DataPipe for applying a transformation function to each element of the input DataPipe.

    This DataPipe allows the user to specify a custom transformation function to modify
    the data elements yielded by the source DataPipe.

    Attributes:
        source_datapipe (IterDataPipe): The source DataPipe providing the data elements.
        transform_fn (Callable): A callable function to apply to each data element.
    """

    def __init__(self, source_datapipe, transform_fn: Callable):
        """
        Initializes the DataPipe with a source DataPipe and a transformation function.

        Args:
            source_datapipe (IterDataPipe): The source DataPipe yielding data elements.
            transform_fn (Callable): The transformation function to apply.
        """
        self.source_datapipe = source_datapipe
        self.transform_fn = transform_fn

    def __iter__(self):
        """
        Iterates over the source DataPipe, applying the transformation function.

        Yields:
            Any: The transformed data element.
        """
        for source in self.source_datapipe:
            yield self.transform_fn(source)


class SpeciesDataPipe(IterDataPipe):
    """
    A DataPipe for processing species data, including loading, shuffling, and batching.

    This DataPipe manages the complete data processing pipeline for species data, including
    loading CSR matrices and metadata, applying transformations, and generating batches.

    Attributes:
        directory_path (str): The directory path containing the data files.
        npz_masks (Union[str, list[str]]): File masks for the CSR matrix files.
        metadata_masks (Union[str, list[str]]): File masks for the metadata files.
        batch_size (int): The size of each batch.
        shuffle (bool): Whether to shuffle the data.
        return_dense (bool): Whether to return dense tensors.
        verbose (bool): If True, enables verbose output.
        transform_fn (Optional[Callable]): A transformation function to apply to the data.
    """

    def __init__(
        self,
        directory_path: str,
        npz_masks: Union[str, list[str]],
        metadata_masks: Union[str, list[str]],
        batch_size: int,
        allow_partials = False,
        shuffle: bool = True,
        return_dense: bool = False,
        verbose: bool = False,
        transform_fn: Optional[Callable] = None,
    ):
        """
        Initializes the SpeciesDataPipe with the specified parameters.

        Args:
            directory_path (str): The directory path containing the data files.
            npz_masks (Union[str, list[str]]): File masks for the CSR matrix files.
            metadata_masks (Union[str, list[str]]): File masks for the metadata files.
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            return_dense (bool): Whether to return dense tensors. Defaults to False.
            verbose (bool): If True, enables verbose output. Defaults to False.
            transform_fn (Optional[Callable]): A transformation function to apply to the data. Defaults to None.
        """
        super(SpeciesDataPipe, self).__init__()

        # Create file lister datapipe for all npz files in dataset
        npz_paths_dp = FileLister(
            root=directory_path,
            masks=npz_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False,
        )

        # Create file lister datapipe for all metadata files
        metadata_paths_dp = FileLister(
            root=directory_path,
            masks=metadata_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False,
        )

        self.zipped_paths_dp = Zipper(npz_paths_dp, metadata_paths_dp)

        # Sanity check that the metadata files and npz files are correlated
        # and all files are masked correctly
        chunk_paths = []
        for npz_path, metadata_path in self.zipped_paths_dp:
            chunk_paths.append(f"\n\t{npz_path}\n\t{metadata_path}")
        if not chunk_paths:
            raise RuntimeError("No files found for masks from file lister")

        if verbose:
            for path in chunk_paths:
                print(path)

        self.batch_size = batch_size
        self.allow_partials = allow_partials
        self.return_dense = return_dense
        self.verbose = verbose
        self._shuffle = shuffle
        self.transform_fn = transform_fn

        dp = self.zipped_paths_dp
        # don't skip workers all load same chunk
        if len(chunk_paths) > 1:
            dp = dp.sharding_filter()

        if self._shuffle:
            dp = dp.shuffle()

        dp = dp.load_matrix_and_dataframe(self.verbose)

        if self._shuffle:
            dp = dp.shuffle_matrix_and_dataframe()

        dp = dp.batch_csr_matrix_and_dataframe(
            self.batch_size, return_dense=self.return_dense, allow_partials=self.allow_partials
        )

        # thought process on removal
        # the matrix is already completely shuffled before batching
        # the shuffle dp holds a buffer and shuffles the buffer by pulling in samples to shuffle
        # this could cause it to hold more npz in memory than desired

        # if self._shuffle:
        #     dp = dp.shuffle()

        if callable(self.transform_fn):
            dp = dp.transform(self.transform_fn)

        if len(chunk_paths) == 1:
            dp = dp.sharding_filter()

        self.dp = dp

    # def _set_seed(self, seed: Optional[int] = None):
    #     if seed is None:
    #         seed = random.randint(0, 1000)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)

    def __iter__(self):
        """
        Iterates over the DataPipe, yielding processed data elements.

        Yields:
            tuple: A tuple containing processed data, typically a sparse tensor and metadata.

        Raises:
            Exception: If there is an error during iteration.
        """
        # self.set_seed()

        try:
            yield from self.dp
        except Exception as e:
            print(f"Error during iteration: {e}")
            raise
        finally:
            # Ensure all resources are properly cleaned up
            pass


class RandomSelectDataPipe(IterDataPipe):
    """
    A DataPipe for randomly selecting data from multiple input DataPipes.
    This DataPipe allows for random selection of data elements from multiple source DataPipes,
    yielding elements until all input DataPipes are exhausted.

    Attributes:
        datapipes (list[IterDataPipe]): A list of source DataPipes to select from.
    """

    def __init__(self, *datapipes):
        """
        Initializes the RandomSelectDataPipe with multiple source DataPipes.

        Args:
            datapipes (list[IterDataPipe]): A list of source DataPipes.
        """
        self.datapipes = datapipes

    def __iter__(self):
        """
        Iterates over the DataPipes, yielding data elements randomly from the available options.

        Yields:
            Any: A randomly selected data element from one of the input DataPipes.

        Raises:
            StopIteration: When all input DataPipes are exhausted.
        """
        iterators = [iter(dp) for dp in self.datapipes]
        while iterators:
            selected_iterator = random.choice(iterators)
            try:
                yield next(selected_iterator)
            except StopIteration:
                # Remove exhausted iterator
                iterators.remove(selected_iterator)
        raise StopIteration


class MultiSpeciesDataPipe(IterDataPipe):
    """
    A DataPipe for handling multiple species data pipelines with selection strategy.
    This DataPipe provides a unified interface to handle multiple species data,
    allowing for random or sequential selection of data elements.

    Attributes:
        datapipe (IterDataPipe): The combined DataPipe for multiple species.
    """

    def __init__(
        self,
        *species: SpeciesDataPipe,
        selection_fn: Union[Literal["random"], Literal["sequential"]] = "random",
    ):
        """
        Initializes the MultiSpeciesDataPipe with species data pipelines and a selection function.

        Args:
            species (list[SpeciesDataPipe]): A list of species DataPipes.
            selection_fn (Union[Literal["random"], Literal["sequential"]]): The selection strategy ('random' or 'sequential'). Defaults to 'random'.
        """
        if selection_fn == "sequential":
            self.datapipe = Multiplexer(*species)
        elif selection_fn == "random":
            self.datapipe = RandomSelectDataPipe(*species)

    def __iter__(self):
        """
        Iterates over the combined species DataPipe, yielding data elements according to the selection strategy.

        Yields:
            Any: A data element from the selected species DataPipe.
        """
        yield from self.datapipe
