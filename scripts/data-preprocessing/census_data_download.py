import argparse as ap
import cellxgene_census
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import tiledbsoma as tdb

from data_processing_functions import normalize_data, save_data_to_disk, verify_data

CENSUS_VERSION = "2024-07-01"
VALUE_FILTER = 'is_primary_data == True' # Metadata filters applied when retrieving data
VALID_SPECIES = ["homo_sapiens", "mus_musculus"]
SPECIES_MAP = {"homo_sapiens": "human", "mus_musculus": "mouse"}

def process_chunk(
    save_dir: os.PathLike,
    species: str,
    ids: list[int],
    chunk_n: int,
):
    """
    Helper function to download and save a chunk of data.

    This is the spawn point of the processes when MultiProcessing
    is used.

    Args:
        save_dir (PathLike):
            Directory to save the data to.
        species (str):
            The name of the species whose data is being downloaded.
            NOTE: The taxonomic name is expected and is later mapped
            to the English name when saving the data.
        ids (list[int]):
            The soma_joinids to query the data that is to be downloaded.
        chunk_n (int):
            Current chunk being processes. Appended to the filename when
            its saved to disk.
    """
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
                                    census= census,
                                    organism= species,
                                    measurement_name= "RNA",
                                    X_name= "raw",
                                    obs_value_filter=VALUE_FILTER,
                                    obs_coords=ids
                                )

    normalize_data(adata.X)
    permutation = np.random.permutation(range(adata.X.shape[0]))
    save_data_to_disk(
        data_path=os.path.join(save_dir, f'{SPECIES_MAP[species]}_counts_{chunk_n}.npz'),
        data=adata.X[permutation, :],
        metadata_path=os.path.join(save_dir, f'{SPECIES_MAP[species]}_metadata_{chunk_n}.pkl'),
        metdata=adata.obs.iloc[permutation].reset_index(drop=True)
    )

def main(
    directory: os.PathLike,
    species: str,
    chunk_size: int,
    processes: int,
    seed: int,
    sample_size: int,
):
    """
    Main downloading function for CellxGene census data.

    Can be used to download all available data that matches the provided
    filters or a random subset.

    Args:
        directory (PathLike):
            The directory to save the downloaded data in.
        species (str):
            The name of the species whose data is being downloaded.
            NOTE: The taxonomic name is expected and is later mapped
            to the English name when saving the data.
        chunk_size (int):
            Number of samples to retrieve and save to disk in a single
            file. Breaks up the total available data into 'chunks'.
        processes (int, optional):
            Number of download processes to run at a time. Helps speed
            up the download but uses more RAM. Defaults to 1.
        seed (int, optional):
            Seed for NumPy's Random module. Set for reproducibility.
            Defaults to 42.
        sample_size (int, optional):
            Number of samples to randomly grab out of the total available
            data. Defaults to None, and all available data is retrieved.
    """
    if species not in VALID_SPECIES:
        raise ValueError(
            f"Error: Invalid species provided - {species}. Valid values are: {VALID_SPECIES}"
        )
    
    if not os.path.exists(directory):
        raise FileExistsError("Error: Provided directory does not exist!")

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:

        soma_ids = census["census_data"][species].obs.read(
                                                    value_filter=VALUE_FILTER,
                                                    column_names=["soma_joinid"]
                                                ).concat().to_pandas()
        soma_ids = list(soma_ids['soma_joinid'])

    total_data = len(soma_ids)
    set_ids = set(soma_ids)
    assert total_data == len(set_ids)
    assert chunk_size <= total_data
    if sample_size is not None:
        assert sample_size <= total_data
        assert chunk_size <= sample_size

    np.random.seed(seed)
    np.random.shuffle(soma_ids)

    if sample_size is not None:
        soma_ids = np.random.choice(soma_ids, sample_size, replace=False)

    mp.set_start_method('spawn')
    with mp.Pool(processes= processes) as pool:
        chunk_count = 0
        for i in range(0, len(soma_ids), chunk_size):
            chunk_count += 1
            pool.apply_async(
                func= process_chunk,
                args=(directory, species, soma_ids[i:i+chunk_size], chunk_count)
            )
        pool.close()
        pool.join()

    verify_data(
        directory= directory,
        species= SPECIES_MAP[species],
        ids= set_ids,
        expected_size= chunk_size,
        last_chunk= chunk_count, 
        last_size= total_data % chunk_size
    )
    
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory to save data in."
    )
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        help="Species to download data for."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Number of samples to save in each chunk of data."
    )
    parser.add_argument(
        "--processes",
        type=int,
        required=False,
        default=1,
        help="Number of sub-processes to use for the download."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Seed for the random module."
    )
    parser.add_argument(
        "--sample_size",
        type=int, 
        required=False, 
        default=None, 
        help="Number of samples to grab out of total. Omit to get all data."
    )

    args = parser.parse_args()

    main(
        directory= args.directory,
        species= args.species,
        chunk_size= args.chunk_size,
        processes= args.processes,
        seed= args.seed,
        sample_size= args.sample_size
    )