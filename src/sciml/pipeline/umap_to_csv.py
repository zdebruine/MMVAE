import os
import numpy as np
import pandas as pd



if __name__ == "__main__":
    
    method = 'random'
    method_dir = os.path.join('/mnt/projects/debruinz_project/integration/bioc/', method)
    embeddings_path = os.path.join(method_dir, "umap", f"{method}_vae_z_embeddings.npz")
    meta_path = os.path.join(method_dir, "integrated_samples", "z_metadata.full.pkl")
    
    embeddings = np.load(embeddings_path)['embeddings']
    df = pd.read_pickle(meta_path)
    df['x'] = embeddings[:, 0]
    df['y'] = embeddings[:, 1]
    
    df.to_csv(os.path.join(method_dir, "umap_embeddings.csv"))