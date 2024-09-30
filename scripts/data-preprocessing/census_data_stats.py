import glob
import os
import argparse as ap
import pandas as pd
import scipy.sparse as sp

from typing import Union
from collections import defaultdict
from data_processing_functions import extract_file_number, gather_stats, record_stats, DATA_CATEGORIES

def update_totals(totals: dict[str, dict[str, int]], data: dict[str, Union[int, dict[str, int]]]):
    
    for category in DATA_CATEGORIES:
        for key, value in data[category]:
            totals[category][key] += value

def main(directory: os.PathLike, species: str):

    data_files = glob.glob(os.path.join(directory, f'{species}*.npz'))
    metadata_files = glob.glob(os.path.join(directory, f'{species}*.pkl'))
    data_files.sort(key=extract_file_number)
    metadata_files.sort(key=extract_file_number)

    totals = {
        cat: defaultdict(int) for cat in DATA_CATEGORIES
    }

    for data_path, metadata_path in zip(data_files, metadata_files):
        data = sp.load_npz(data_path)
        metadata = pd.read_pickle(metadata_path)

        chunk_stats = gather_stats(data, metadata)
        filename = f'{os.path.basename(data_path).split(".")[0]}_stats.csv'

        update_totals(totals, chunk_stats)

        record_stats(
            path= os.path.join(directory, filename),
            data= chunk_stats
        )

    record_stats(
        path=os.path.join(directory, f'{species}_stat_totals.csv'),
        data=totals
    )

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory to load data from.")
    parser.add_argument("--species", type=str, required=True, help="Species to check data for.")

    args = parser.parse_args()
    main(
        directory= args.directory,
        species= args.species
    )