import os
import re
import argparse
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

def merge_predictions_main(args):
    if not args.regex:
        args.regex = r'z_(embeddings_\d+\.npz|metadata_\d+\.pkl)'
        
    if not args.save_path:
        args.save_path = args.root_dir
    
    files = get_matching_files(args.root_dir, args.regex)
    
    embedding_files = []
    metadata_files = []

    embedding_pattern = re.compile(r'z_embeddings_(\d+)\.npz')
    metadata_pattern = re.compile(r'z_metadata_(\d+)\.pkl')

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
    
    np.savez(f"{args.save_path}/z_embeddings.full.npz", embeddings=embeddings)
    metadata.to_pickle(f"{args.save_path}/z_metadata.full.pkl")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Merge Predictions")
    parser.add_argument('-d', '--root_dir', type=str, required=True, help="Directory where samples are stored to be merged")
    parser.add_argument('-s', '--save_path', type=str, help="Path to aggreated file location")
    parser.add_argument('-r', '--regex', type=str, help="Regex to identify pairings")
    args = parser.parse_args()
    assert os.path.exists(args.root_dir), f"Root directory does not exist {args.root_dir}"
    merge_predictions_main(args)
