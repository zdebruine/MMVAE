import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
    
def plot_umap(
    npz_path, 
    meta_path, 
    save_path, 
    n_neighbors = 30,
    min_dist = 0.3,
    n_components = 2,
    metric = "cosine",
    low_memory = False,
    n_jobs = 40,
    n_epochs = 200,
    n_largest = 15,
    categories = ['cell_type', 'dataset_id', 'assay'],
    **umap_kwargs,
):
    
    X = np.load(npz_path)['arr_0']
    metadata = pd.read_pickle(meta_path)
    
    # Fit and transform the data using UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        n_components=n_components,
        metric=metric, 
        low_memory=low_memory, 
        n_jobs=n_jobs, 
        n_epochs=n_epochs, 
        **umap_kwargs)
    
    embedding = reducer.fit_transform(X)
    
    for category in categories:
        plot_category(embedding, metadata, category, save_path, n_largest)
        

def plot_category(embedding, metadata, category, save_path, n_largest):
    plt.figure(figsize=(14, 8))
    unique_values = metadata[category].value_counts().nlargest(n_largest).index
    
    # Prepare color map and markers
    cmap = plt.get_cmap('nipy_spectral', len(unique_values))
    color_list = [cmap(i) for i in range(len(unique_values))]
    markers = {}

    # Plotting each category
    for i, value in enumerate(unique_values):
        idx = metadata[category] == value
        scatter = plt.scatter(embedding[idx, 0], embedding[idx, 1], color=color_list[i], label=value, s=5)
        markers[value] = scatter
    
    plt.title(f'UMAP projection colored by {category}')
    
    # Custom legend with a circle for each label
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=label)
                      for i, label in enumerate(unique_values)]
    plt.legend(handles=legend_handles, title=category, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.savefig(f"{save_path}_{category}.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='UMAP Projection Plotting')
    parser.add_argument('--npz_path', type=str, required=True, help="Path to npz file containing the feature matrix")
    parser.add_argument('--meta_path', type=str, required=True, help="Path to metadata pickle file")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the resulting plots")
    args = parser.parse_args() 
    
    plot_umap(args.npz_path, args.meta_path, args.save_path)
