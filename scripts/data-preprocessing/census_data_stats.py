import glob
import os
import argparse as ap
from collections import defaultdict

import pandas as pd
import scipy.sparse as sp

from data_processing_functions import extract_file_number, gather_stats, record_stats, DATA_CATEGORIES

def main(directory: os.PathLike, species: str):
    """
    Main function for gathering stats on downloaded CellxGene Census data.

    The mean and standard deviation for each file are recorded to a csv.
    The counts for each unique value for each type of data specified in 
    DATA_CATEGORIES is also recorded per-file alongside total counts.

    The is meant as a sanity check on randomly sampled subsets of the
    full data and is not intended to gather stats on all available 
    CellxGene census data.

    Args:
        directory (PathLike):
            The directory holding the downloaded data.
        species (str):
            The name of the species whose data is being checked. This
            is used to perform pattern matching in the directory.
    """
    data_files = glob.glob(os.path.join(directory, f"{species}*.npz"))
    metadata_files = glob.glob(os.path.join(directory, f"{species}*.pkl"))

    # Sort by file number rather than filename
    data_files.sort(key=extract_file_number)
    metadata_files.sort(key=extract_file_number)

    # Storage for data on all files
    count_stats = defaultdict(dict)
    total_stats = {
        category: defaultdict(lambda: defaultdict(int))
        for category in DATA_CATEGORIES
    }
    filenames = []

    for data_path, metadata_path in zip(data_files, metadata_files):

        file = os.path.basename(data_path).split(".")[0]
        filenames.append(file)
        
        data = sp.load_npz(data_path)
        metadata = pd.read_pickle(metadata_path)
        
        chunk_stats = gather_stats(data, metadata)

        count_stats["mean"][file] = chunk_stats["mean"]
        count_stats["std"][file] = chunk_stats["std"]

        # Update the total counts with the current files data
        for category in DATA_CATEGORIES:
            for value, count in chunk_stats[category].items():
                total_stats[category][value][file] = count

    features_df = pd.DataFrame(count_stats)
    features_df.to_csv(
        os.path.join(directory,
        f"{species}_data_distribution.csv")
    )

    for category in DATA_CATEGORIES:
        df = pd.DataFrame(data[category].T.fillna(0))
        df.columns = filenames
        df["Total"] = df.sum(axis=1)
        df.to_csv(os.path.join(directory, f"{species}_{category}_distribution.csv"))

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory to load data from."
    )
    parser.add_argument(
        "--species", type=str, required=True, help="Species to check data for."
    )

    args = parser.parse_args()
    main(
        directory= args.directory,
        species= args.species
    )