import argparse as ap
import cellxgene_census
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import tiledbsoma as tdb

from data_processing_functions import normalize_data, save_data_to_disk, verify_data

VALUE_FILTER = 'is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]'
VALID_SPECIES = ["homo_sapiens", "mus_musculus"]
SPECIES_MAP = {"homo_sapiens": "human", "mus_musculus": "mouse"}

def process_chunk(census: tdb.Collection, species: str, ids: list[int], chunk_n: int, save_dir: os.PathLike):
    adata = cellxgene_census.get_anndata(census, species, "RNA", "raw", obs_value_filter=VALUE_FILTER, obs_coords=ids)
    normalize_data(adata.X)
    permutation = np.random.permutation(range(adata.X.shape[0]))
    save_data_to_disk(
        data_path=os.path.join(save_dir, f'{SPECIES_MAP[species]}_counts_{chunk_n}.npz'),
        data=adata.X[permutation, :],
        metadata_path=os.path.join(save_dir, f'{SPECIES_MAP[species]}_metadata_{chunk_n}.pkl'),
        metdata=adata.obs.iloc[permutation].reset_index(drop=True)
    )

def main(directory: os.PathLike, species: str, chunk_size: int, seed: int, sample_size: int):

    if species not in VALID_SPECIES:
        raise ValueError(f"Error: Invalid species provided - {species}. Valid values are: {VALID_SPECIES}")
    
    if not os.path.exists(directory):
        raise FileExistsError("Error: Provided directory does not exist!")

    with cellxgene_census.open_soma(census_version="2024-07-01") as census:

        soma_ids = census["census_data"][species].obs.read(value_filter=VALUE_FILTER, column_names=["soma_joinid"]).concat().to_pandas()
        soma_ids = list(soma_ids['soma_joinid'])

        num_samples = len(soma_ids)
        set_ids = set(soma_ids)
        assert num_samples == len(set_ids)

        np.random.seed(seed)
        np.random.shuffle(soma_ids)

        if sample_size is not None:
            soma_ids = np.random.choice(soma_ids, sample_size, replace=False)

        chunk_count = 0
        for i in range(0, len(soma_ids), chunk_size):
            chunk_count += 1
            process_chunk(census, species, soma_ids[i:i+chunk_size], chunk_count, directory)

    verify_data(directory, SPECIES_MAP[species], set_ids, chunk_size, chunk_count, num_samples % chunk_size)
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory to save data in.")
    parser.add_argument("--species", type=str, required=True, help="Species to download data for.")
    parser.add_argument("--chunk_size", type=int, required=True, help="Number of samples to save in each chunk of data.")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Seed for the random module")
    parser.add_argument("--sample_size", type=int, required=False, default=None, help="Number of samples to grab out of total. Omit to get all data.")

    args = parser.parse_args()

    main(
        directory= args.directory,
        species= args.species,
        chunk_size= args.chunk_size,
        seed= args.seed,
        sample_size= args.sample_size
    )