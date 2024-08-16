"""
Generate UMAPs from embeddings and metadata.
"""

import os
import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import click


def load_embeddings(npz_path, meta_path):
    """Load embeddings and metadata from specified paths."""
    embedding = np.load(npz_path)['embeddings']
    metadata = pd.read_pickle(meta_path)
    return embedding, metadata

def plot_umap(
    directory,
    keys,
    categories,
    n_neighbors=30,
    min_dist=0.3,
    n_components=2,
    metric="cosine",
    low_memory=False,
    n_jobs=40,
    n_epochs=200,
    n_largest=15,
    method=None,
    save_dir=None,
    **umap_kwargs,
):
    """
    Generate UMAP embeddings and plot them.

    Args:
        directory (str): Directory where embeddings are stored.
        keys (list[str]): List of embedding keys.
        categories (list[str]): List of categories to color by.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.
        n_components (int): Number of components for UMAP.
        metric (str): Metric for UMAP.
        low_memory (bool): Low memory setting for UMAP.
        n_jobs (int): Number of CPUs available for UMAP.
        n_epochs (int): Number of epochs to run UMAP.
        n_largest (int): Number of most common categories to plot.
        method (str): Method title to add to the graph.
        save_dir (str): Directory to save UMAP plots.
        **umap_kwargs: Extra kwargs passed to `umap.UMAP`.
    """
    if not save_dir:
        save_dir = directory
        
    image_paths = []
    for key in keys:
        umap_path = os.path.join(directory, f"{key}_umap_embeddings.npz")
        if os.path.exists(umap_path):
            npz_path = umap_path
            meta_path = os.path.join(directory, f"{key}_umap_metadata.pkl")
            embedding, metadata = load_embeddings(npz_path, meta_path)
        else:
            npz_path = os.path.join(directory, f"{key}_embeddings.npz")
            meta_path = os.path.join(directory, f"{key}_metadata.pkl")
            X, metadata = load_embeddings(npz_path, meta_path)
            
            # Fit and transform the data using UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                low_memory=low_memory,
                n_jobs=n_jobs,
                n_epochs=n_epochs,
                **umap_kwargs
            )
            
            embedding = reducer.fit_transform(X)
            embedding_path = os.path.join(save_dir, f'{key}_umap_embeddings.npz')
            metadata_path = os.path.join(save_dir, f'{key}_umap_metadata.pkl')
            os.makedirs(save_dir, exist_ok=True)
            np.savez(embedding_path, embeddings=embedding)
            metadata.to_pickle(metadata_path)
        
        for category in categories:
            image_path = plot_category(embedding, metadata, category, save_dir, n_largest, key, method)
            image_paths.append(image_path)
    return image_paths

def plot_category(embedding, metadata, category, save_path, n_largest, name, method, alpha=0.5, marker_size=1):
    """
    Plot UMAP embeddings colored by a specific category.

    Args:
        embedding (np.ndarray): The UMAP embeddings.
        metadata (pd.DataFrame): The metadata associated with embeddings.
        category (str): Category to color by.
        save_path (str): Directory to save the plot.
        n_largest (int): Number of most common categories to plot.
        name (str): Name for the file.
        method (str): Method title to add to the graph.
        alpha (float): Opacity of plot points.
        marker_size (int): Size of plot points.

    Returns:
        str: Path to the saved plot image.
    """
    plt.figure(figsize=(14, 8))
    unique_values = metadata[category].value_counts().nlargest(n_largest).index
    
    # Prepare color map
    cmap = plt.get_cmap('nipy_spectral', len(unique_values))
    color_list = [cmap(i) for i in range(len(unique_values))]
    
    # Combine embedding and metadata into a DataFrame
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    df[category] = metadata[category].values
    
    # Filter to include only the largest categories
    df = df[df[category].isin(unique_values)]
    
    # Shuffle the DataFrame to randomize the plotting order
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Create a dictionary to map categories to colors
    category_to_color = {value: color_list[i] for i, value in enumerate(unique_values)}
    
    # Map colors to the entire DataFrame
    df['color'] = df[category].map(category_to_color)
    
    # Plot all points in the shuffled order with specified opacity and marker size
    plt.scatter(df['x'], df['y'], c=df['color'], s=marker_size, alpha=alpha)
    
    if method:
        method_str = f" for {method} "
    else:
        method_str = " "
    
    plt.title(f'UMAP projection{method_str}colored by {category}')
    
    # Custom legend with a circle for each label
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=label)
                      for i, label in enumerate(unique_values)]
    plt.legend(handles=legend_handles, title=category, bbox_to_anchor=(1.05, 1), loc='upper left')
    image_path = os.path.join(save_path, f"integrated.{category}.umap.{name}.png")
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    return image_path

def add_images_to_tensorboard(log_dir, image_paths):
    """
    Add images to Tensorboard.

    Args:
        log_dir (str): Path to Tensorboard log directory.
        image_paths (list[str]): Paths to images to load to Tensorboard.
    """
    writer = SummaryWriter(log_dir=log_dir)
    
    for image_path in image_paths:
        image = Image.open(image_path)
        image = np.array(image)
        image = torch.tensor(image).permute(2, 0, 1)
        tag = os.path.basename(image_path)
        writer.add_image(tag, image, global_step=0)
    writer.close()

def generate_umap(
    directory: str,
    categories: tuple[str],
    keys: tuple[str],
    method: str,
    save_dir: str,
    skip_tensorboard: bool,
):
    image_paths = plot_umap(directory=directory, keys=keys, categories=categories, method=method, save_dir=save_dir)

    if not skip_tensorboard:
        add_images_to_tensorboard(directory, image_paths)


@click.command(name='generate_umap')
@click.option('--directory', type=click.Path(exists=True), required=True)
@click.option('--category', type=str, multiple=True, required=True)  # Multiple positional arguments
@click.option('--key', type=str, multiple=True, required=True) 
@click.option('--save_dir', type=click.Path(), help="Directory to store PNGs")
@click.option('--method', type=str, help="Method name to add to graph title")
@click.option('--skip_tensorboard', is_flag=True, help="Prevent logging UMAPs to Tensorboard")
def main(directory, category, key, method, save_dir, skip_tensorboard):
    """
    Plot UMAP embeddings and optionally log images to Tensorboard.

    Args:
        directory (str): Directory where embeddings to plot are stored.
        categories (tuple[str]): List of categories to color by.
        keys (tuple[str]): List of embedding keys that prefix save_paths of _embeddings.npz and _metadata.pkl
        save_dir (str): Path to save UMAP outputs.
        skip_tensorboard (bool): Prevent logging UMAPs to Tensorboard.
    """
    generate_umap(directory, category, key, method, save_dir, skip_tensorboard)

if __name__ == "__main__":
    main()
