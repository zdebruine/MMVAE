"""
    Merge Embeddings and Metadata.
    
    This module epects files to be stored by {key}_embddings_{\d}.npz and {key}_metadata_{\d}.pkl
    and outputs a single {key}_embedding.npz and {key}_metadata.pkl for each key.
"""
import os
import re
import argparse
import sys
import numpy as np
import pandas as pd


def get_matching_files(directory, pattern):
    pattern = re.compile(pattern)
    matching_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def extract_index(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return -1

def merge_predictions_main(
    directory: str,
    save_dir: str,
    keys: list[str]
):
    """
    Merge's saved embeddings and metadata into one npz and pkl file.
    
    Args:
        directory (str): Directory where _embeddings{\d}.npz and _metadata{\d}.pkl are stored.
        save_dir (str): Directory to store {key}_embeddings.npz and {key}_metadata.npz
        keys (list[str]): List of prefix keys for embeddings and metadata paths.
    """
    assert os.path.exists(directory), f"Root directory does not exist {directory}"
    for key in keys:
        regex = rf"{key}(_embeddings_\d+\.npz|_metadata_\d+\.pkl)"
            
        if not save_dir:
            save_dir = directory
        
        files = get_matching_files(directory, regex)
        
        embedding_files = []
        metadata_files = []

        embedding_pattern = re.compile(rf'{key}_embeddings_(\d+)\.npz')
        metadata_pattern = re.compile(rf'{key}_metadata_(\d+)\.pkl')

        for file in files:
            if embedding_pattern.search(file):
                embedding_files.append(file)
            elif metadata_pattern.search(file):
                metadata_files.append(file)
                
        if not embedding_files or not metadata_files:
            raise FileNotFoundError(f"No files found! {files}, \nembeddings: {embedding_files} \nmetadata:{metadata_files}")
        
        embedding_files.sort(key=lambda x: extract_index(x, embedding_pattern))
        metadata_files.sort(key=lambda x: extract_index(x, metadata_pattern))
        
        embeddings = np.concatenate([np.load(file)['embeddings'] for file in embedding_files])
        metadata = pd.concat([pd.read_pickle(file) for file in metadata_files])
        
        np.savez(f"{save_dir}/{key}_embeddings.npz", embeddings=embeddings)
        metadata.to_pickle(f"{save_dir}/{key}_metadata.pkl")

def main():
    parser = argparse.ArgumentParser(description="Merge Predictions")
    parser.add_argument('-d', '--directory', required=True, help="Directory where samples are stored to be merged")
    parser.add_argument('-s', '--save_dir', required=True, help="Path to aggregated file location")
    parser.add_argument('--keys', nargs='*', required=True, help="Keys to merge")
    args = parser.parse_args()
    # Call the merge_predictions_main function with parsed arguments
    merge_predictions_main(
        directory=args.directory,
        save_dir=args.save_dir,
        keys=args.keys
    )

if __name__ == "__main__":
    main()