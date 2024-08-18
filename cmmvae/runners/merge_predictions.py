"""
Merge Embeddings and Metadata.

This module expects files to be stored by {key}_embeddings_{\d}.npz and {key}_metadata_{\d}.pkl
and outputs a single {key}_embedding.npz and {key}_metadata.pkl for each key.
"""

import os
import re
import click
import numpy as np
import pandas as pd
from ._decorators import click_env_option


def get_matching_files(directory, pattern):
    """Return a list of files matching the given pattern in the directory."""
    pattern = re.compile(pattern)
    matching_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def extract_index(filename, pattern):
    """Extract numeric index from the filename based on the pattern."""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return -1

def merge_predictions(directory, save_dir, keys):

    assert os.path.exists(directory), f"Root directory does not exist: {directory}"
    
    if not save_dir:
        save_dir = directory

    for key in keys:
        regex = rf"{key}(_embeddings_\d+\.npz|_metadata_\d+\.pkl)"
        
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
            raise FileNotFoundError(f"No files found for key '{key}'.\nEmbeddings: {embedding_files}\nMetadata: {metadata_files}")
        
        embedding_files.sort(key=lambda x: extract_index(x, embedding_pattern))
        metadata_files.sort(key=lambda x: extract_index(x, metadata_pattern))
        
        embeddings = np.concatenate([np.load(file)['embeddings'] for file in embedding_files])
        metadata = pd.concat([pd.read_pickle(file) for file in metadata_files])
        
        np.savez(os.path.join(save_dir, f"{key}_embeddings.npz"), embeddings=embeddings)
        metadata.to_pickle(os.path.join(save_dir, f"{key}_metadata.pkl"))


@click.command(name="merge_predictions")
@click_env_option('--directory', type=click.Path(exists=True), required=True, help="Path to the directory containing the embeddings and metadata.")
@click_env_option('--save_dir', type=click.Path(), default=None, help="Directory to store merged predictions. Defaults to the same as the input directory if not provided.")
@click_env_option('--keys', multiple=True, required=True, help="List of prefix keys for embeddings and metadata paths.")
def merge_predictions(directory, save_dir, keys):
    """
    Merge saved embeddings and metadata into one npz and pkl file.

    Args:
        directory (str): Directory where _embeddings{\d}.npz and _metadata{\d}.pkl are stored.
        save_dir (str): Directory to store {key}_embeddings.npz and {key}_metadata.pkl. Defaults to directory if not provided.
        keys (list[str]): List of prefix keys for embeddings and metadata paths.
    """
    merge_predictions(directory, save_dir, keys)
    
def cli():
    merge_predictions()

if __name__ == '__main__':
    cli()
