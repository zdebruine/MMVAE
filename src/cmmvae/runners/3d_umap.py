import os
import numpy as np
import pandas as pd
import pickle
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def load_embeddings(npz_path, meta_path):
    """Load embeddings and metadata from specified paths."""
    embedding = np.load(npz_path)["embeddings"]
    metadata = pd.read_pickle(meta_path)
    return embedding, metadata


def plot_3d_umap(
    embedding, metadata, column, output_path, fig_size=(10, 10), point_size=2
):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=metadata[column].astype("category").cat.codes,
        cmap="hsv",
        s=point_size,
    )
    ax.set_title(
        f'3D UMAP projection colored by {column}\nFilename: {output_path.split("/")[-1]}'
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    def update(num):
        ax.view_init(elev=30, azim=num)

    ani = FuncAnimation(fig, update, frames=360, interval=20)
    ani.save(output_path.replace(".png", ".gif"), writer="imagemagick")

    plt.close()


def plot_umap(
    X,
    metadata,
    n_neighbors=30,
    min_dist=0.3,
    n_components=3,
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

    # Fit and transform the data using UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        low_memory=low_memory,
        n_jobs=n_jobs,
        n_epochs=n_epochs,
        **umap_kwargs,
    )

    embedding = reducer.fit_transform(X)
    embedding_file_name = "3d_z_umap_embeddings.npz"
    embedding_path = os.path.join(save_dir, embedding_file_name)
    metadata_file_name = "3d_z_umap_metadata.pkl"
    metadata_path = os.path.join(save_dir, metadata_file_name)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(embedding_path, embeddings=embedding)
    metadata.to_pickle(metadata_path)


directory = "/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/baseline/1e7Beta_256rclt_noadv/"
embedding, metadata = load_embeddings(
    npz_path=os.path.join(directory, "merged/z_embeddings.npz"),
    meta_path=os.path.join(directory, "merged/z_metadata.pkl"),
)

plot_umap(embedding, metadata, save_dir=os.path.join(directory, "umap"))

embedding, metadata = load_embeddings(
    npz_path=os.path.join(directory, "umap/3d_z_umap_embeddings.npz"),
    meta_path=os.path.join(directory, "umap/3d_z_umap_metadata.pkl"),
)
categories = ["donor_id", "assay", "dataset_id", "cell_type", "tissue", "species"]
for col in categories:
    plot_3d_umap(
        embedding=embedding,
        metadata=metadata,
        column=col,
        output_path=os.path.join(directory, f"umap/3D_UMAP_by_{col}.png"),
    )
