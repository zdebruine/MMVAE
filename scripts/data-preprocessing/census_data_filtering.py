import os
import re
import glob
import argparse as ap
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import data_filtering_functions as ff
from collections import defaultdict
from data_processing_functions import extract_file_number

def main(
        directory: str,
        species: list[str],
        train_dir: str,
        save_dir: str,
        seed: int,
        skip_metadata: bool,
):
    if not skip_metadata:
        np.random.seed(seed)
        dataframes = {}
        for specie in species:
            # Process file paths for full and training data
            train_files = tuple(glob.glob(os.path.join(train_dir, f"{specie}*.pkl")))
            train_files = ff.filter_and_sort_train_files(train_files)

            # Get somaIDs of training samples
            train_ids = ff.get_train_data_ids(train_files)

            # Prepare main df that is to be filtered
            metadata_files = glob.glob(os.path.join(directory, f"{specie}*.pkl"))
            metadata_files.sort(key=extract_file_number)
            meta_dataframe = ff.load_and_merge_metadata(tuple(metadata_files))

            # Filter out training samples from the full data
            dataframes[specie] = ff.filter_train_ids(meta_dataframe, train_ids)

        # Process groups and subset size if needed
        grouped_data = ff.filter_into_groups(dataframes)
        valid_groups = ff.validate_and_sample_groups(grouped_data, species[0])

        # Save filtered metadata
        ff.save_grouped_data(valid_groups, dataframes, save_dir)

    # Load data and slice by metadata index
    for specie in species:
        data_files = glob.glob(os.path.join(directory, f"{specie}*.npz"))
        data_files.sort(key=extract_file_number)
        metadata_files = glob.glob(os.path.join(save_dir, f"{specie}*.pkl"))
        metadata_files.sort(key=extract_file_number)
        filtered_data = defaultdict(list)
        for chunk_n, data_file in enumerate(data_files, start=1):
            current_chunk = sp.load_npz(data_file)
            for gid, metadata_file in enumerate(metadata_files, start=1):
                current_df = pd.read_pickle(metadata_file)
                idxes = current_df[current_df["chunk_source"] == chunk_n]["data_index"]
                sliced_chunk = current_chunk[idxes, :]
                filtered_data[gid].append(sliced_chunk)
                
        # Save filtered data
        for gid, data in filtered_data.items():
            chunk_data = sp.vstack(data)
            sp.save_npz(os.path.join(save_dir, f"{specie}_filtered_{gid}.npz"), chunk_data)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory to load data from."
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        required=True,
        help="Species to load data for."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=False,
        default=None,
        help="Directory where the training data is stored. Defaults to '--directory'"
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        required=False,
        default=None,
        help="Directory to save filtered data in. Defaults to '--directory'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Seed for the random module."
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        help="Whether to skip the metadata filtering and just slice the count data."
    )
    
    args = parser.parse_args()

    train_dir = args.directory if args.train_data is None else args.train_data
    save_dir = args.directory if args.save_directory is None else args.save_directory

    main(
        directory= args.directory,
        species= args.species,
        train_dir= train_dir,
        save_dir= save_dir,
        seed= args.seed,
        skip_metadata= args.skip_metadata
    )