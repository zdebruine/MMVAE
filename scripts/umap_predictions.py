import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp

def plot_umap(args):
    
    X = None
    Y = None
    with open(args.npz_path, 'rb') as file:
        X = sp.load_npz(file)
        
    with open(args.meta_path, 'rb') as file:
        Y = pd.read_csv(file)
    import umap
    
    # Fit and transform the data using UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2, metric="cosine")
    embedding = reducer.fit_transform(X)

    #sp.save_npz(f"{args.save_path}.embedding", embedding)

    # Plot the results
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='viridis', s=5)
    plt.colorbar()
    plt.title('UMAP projection of the Digits dataset')
    plt.savefig(args.save_path)
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser('UMAP')
    parser.add_argument('--npz_path', type=str, help="Path to npz file to load")
    parser.add_argument('--meta_path', type=str, help="Path to metadata")
    parser.add_argument('--save_path', type=str, help="Path to save plt figure")
    
    args = parser.parse_args()
    
    plot_umap(args)
