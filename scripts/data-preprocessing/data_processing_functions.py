import csv
import glob
import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

from typing import Union

DATA_CATEGORIES = ['assay', 'cell_type', 'tissue']

def normalize_data(data: sp.csr_matrix):
    for row in range(data.shape[0]):
        start_idx = data.indptr[row]
        end_idx = data.indptr[row + 1]

        row_data = data.getrow(row).data
        row_sum = row_data.sum()

        # Apply the transformation and log1p
        data.data[start_idx:end_idx] = np.log1p((row_data * 1e4) / row_sum)


def save_data_to_disk(data_path: os.PathLike, metadata_path: os.PathLike, data: sp.csr_matrix, metdata: pd.DataFrame):
    sp.save_npz(data_path, data)
    metdata.to_pickle(metadata_path, compression=None)

def record_stats(path: os.PathLike, data: dict[str, Union[int, dict[str, int]]]):
    
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Stat', 'Value(s)'])

        for key, value in data.items():
            if isinstance(value, dict):
                merged = ', '.join([f'{k} - {v}' for k, v in value.items()])
            else:
                writer.writerow([key, value])

def get_data_stats(data: sp.csr_matrix):
    row_sums = data.sum(axis=1).A.flatten()
    return row_sums.mean(), np.std(row_sums)

def gather_stats(data: sp.csr_matrix, metadata: pd.DataFrame):
    chunk_stats = {}
    mean, std = get_data_stats(data)
    chunk_stats["mean"] = mean
    chunk_stats["std"] = std

    for col in DATA_CATEGORIES:
        chunk_stats[col] = metadata[col].value_counts().to_dict()

    return chunk_stats

def extract_file_number(filename):
        match = re.search(r'_(\d+)', filename)  # Look for digits after an underscore (_)
        return int(match.group(1)) if match else 0 

def verify_data(directory: os.PathLike, species: str, ids: set[int], expected_size: int, last_chunk: int = None, last_size: int = None):
    
    data_files = glob.glob(os.path.join(directory, f'{species}*.npz'))
    metadata_files = glob.glob(os.path.join(directory, f'{species}*.pkl'))
    data_files.sort(key=extract_file_number)
    metadata_files.sort(key=extract_file_number)

    for i, (data_path, metadata_path) in enumerate(zip(data_files, metadata_files), start=1):
        errors_detected = False

        data = sp.load_npz(data_path)
        metadata = pd.read_pickle(metadata_path)

        if data.shape[0] != expected_size:
            if i != last_chunk:
                print(f"Chunk size mismatch in chunk #{i}!!!!!")
                errors_detected = True
            elif data.shape[0] != last_size:
                print(f"Chunk size mismatch in chunk #{i}!!!!!")
                errors_detected = True
        
        soma_ids = list(metadata['soma_joinid'])
        num_ids = len(soma_ids)
        if num_ids != data.shape[0]:
            print(f"Metadata mismatch in chunk #{i}!!!!!")
            errors_detected = True

        set_ids = set(soma_ids)
        if len(set_ids) != num_ids:
            print(f'Duplicate IDs found in chunk #{i}!!!!!')
            errors_detected = True

        ids.intersection_update(set_ids)

        if not errors_detected:
            print(f"No issues found in Chunk #{i}.")