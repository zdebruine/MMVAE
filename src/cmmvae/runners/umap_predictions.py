"""
Generate UMAPs from embeddings and metadata.
"""
from typing import Optional
import os
import umap
import numpy as np
import pandas as pd
import click
import h5py
from pathlib import Path

from cmmvae.callbacks.prediction_writer import load_from_hdf5


def load_embeddings(npz_path, meta_path):
    """Load embeddings and metadata from specified paths."""
    embedding = np.load(npz_path)["embeddings"]
    metadata = pd.read_pickle(meta_path)
    return embedding, metadata


def umap_embeddings(
    X,
    n_neighbors=30,
    min_dist=0.3,
    n_components=2,
    metric="cosine",
    low_memory=False,
    n_jobs=40,
    n_epochs=200,
    **kwargs,
):
    print("Fitting umap embeddings...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        low_memory=low_memory,
        n_jobs=n_jobs,
        n_epochs=n_epochs,
        **kwargs,
    )

    embedding = reducer.fit_transform(X)
    print("Done fitting umap embeddings.")
    return embedding


def plot_umap(
    directory,
    categories,
    keys,
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
            embedding = umap_embeddings(X)
            embedding_file_name = f"{key}_umap_embeddings.npz"
            embedding_path = os.path.join(save_dir, embedding_file_name)
            metadata_file_name = f"{key}_umap_metadata.pkl"
            metadata_path = os.path.join(save_dir, metadata_file_name)
            os.makedirs(save_dir, exist_ok=True)
            np.savez(embedding_path, embeddings=embedding)
            metadata.to_pickle(metadata_path)

        for category in categories:
            image_path = plot_category(
                embedding, metadata, category, save_dir, n_largest, key, method
            )
            image_paths.append(image_path)
    return image_paths


def plot_umap_h5(
    hdf5_filepath: str,
    keys,
    categories,
    save_dir=None,
    n_largest=15,
    method="",
    **kwargs,
):
    if not os.path.exists(hdf5_filepath):
        raise FileNotFoundError(hdf5_filepath)

    image_paths = []
    for key in keys:
        data, metadata, embeddings = load_from_hdf5(hdf5_filepath, key)

        if embeddings is None:
            embeddings = umap_embeddings(data)
            with h5py.File(hdf5_filepath, "a") as h5file:
                if f"{key}/umap_embeddings" in h5file:
                    del h5file[f"{key}/umap_embeddings"]
                h5file.create_dataset(
                    f"{key}/umap_embeddings",
                    data=embeddings,
                    shape=embeddings.shape,
                    maxshape=embeddings.shape,
                    dtype=embeddings.dtype,
                )

        save_dir = os.path.dirname(hdf5_filepath) if not save_dir else save_dir
        print(f"Plotting cateogrys for key {key}")
        image_paths.extend(
            [
                plot_category(
                    embeddings, metadata, category, save_dir, n_largest, key, method
                )
                for category in categories
            ]
        )
    print(f"Plotted images at {image_paths}")
    return image_paths


def plot_category(
    embedding,
    metadata,
    category,
    save_path,
    n_largest,
    name,
    method,
    alpha=0.5,
    marker_size=1,
) -> str:
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
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))
    unique_values = metadata[category].value_counts().nlargest(n_largest).index

    # Prepare color map
    cmap = plt.get_cmap("nipy_spectral", len(unique_values))
    color_list = [cmap(i) for i in range(len(unique_values))]

    # Combine embedding and metadata into a DataFrame
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df[category] = metadata[category].values

    # Filter to include only the largest categories
    df = df[df[category].isin(unique_values)]

    # Shuffle the DataFrame to randomize the plotting order
    df = df.sample(frac=1).reset_index(drop=True)

    # Create a dictionary to map categories to colors
    category_to_color = {value: color_list[i] for i, value in enumerate(unique_values)}

    # Map colors to the entire DataFrame
    df["color"] = df[category].map(category_to_color)

    # Plot all points in the shuffled order
    # with specified opacity and marker size
    plt.scatter(x=df["x"], y=df["y"], c=df["color"], s=marker_size, alpha=alpha)

    if method:
        method_str = f" for {method} "
    else:
        method_str = " "

    plt.title(f"UMAP projection{method_str}colored by {category}")

    # Custom legend with a circle for each label
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=cmap(i),
            markersize=10,
        )
        for i, label in enumerate(unique_values)
    ]

    plt.legend(
        handles=legend_handles,
        title=category,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    image_file_name = f"integrated.{category}.umap.{name}.png"
    image_path = os.path.join(save_path, image_file_name)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()
    return image_path


def add_images_to_tensorboard(
    tensorboard_dir,
    image_paths,
    tag="",
):
    """
    Add images to Tensorboard.

    Args:
        log_dir (str): Path to Tensorboard log directory.
        image_paths (list[str]): Paths to images to load to Tensorboard.
    """
    from torch.utils.tensorboard.writer import SummaryWriter
    from PIL import Image
    from torch import tensor

    log_dir = Path(tensorboard_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for image_path in image_paths:
        image_path = Path(image_path)
        image = Image.open(image_path)
        image = np.array(image)
        image = tensor(image).permute(2, 0, 1)
        if not tag:
            path = os.path.normpath(image_path)
            components = path.split(os.sep)
            tag = os.path.join(*components[-2:])
        print(f"Adding image to tensorboard -- {image_path}: {tag}")
        writer.add_image(tag, image, global_step=0)
    writer.close()


def generate_umap(
    directory: str,
    categories: tuple[str],
    keys: tuple[str],
    method: str = "",
    save_dir: Optional[str] = None,
    skip_tensorboard: bool = True,
):
    if directory.endswith(".h5"):
        image_paths = plot_umap_h5(
            hdf5_filepath=directory, keys=keys, categories=categories, save_dir=save_dir
        )
        directory = os.path.dirname(directory)
    else:
        image_paths = plot_umap(
            directory=directory,
            keys=keys,
            categories=categories,
            method=method,
            save_dir=save_dir,
        )

    if not skip_tensorboard:
        add_images_to_tensorboard(directory, image_paths)


@click.command()
@click.option(
    "--directory",
    type=click.Path(exists=True),
    required=True,
    default=lambda: os.environ.get("DIRECTORY", ""),
    help="Directory where embeddings and metadata stored.",
)
@click.option(
    "--categories",
    type=str,
    multiple=True,
    required=True,
    help="Categories to color by.",
)  # Multiple positional arguments
@click.option(
    "--keys",
    type=str,
    multiple=True,
    required=True,
    help="Keys that prefix the embeddings and metadata.",
)
@click.option("--save_dir", type=click.Path(), help="Directory to store PNGs")
@click.option("--method", type=str, help="Method name to add to graph title")
@click.option(
    "--skip_tensorboard", is_flag=True, help="Prevent logging UMAPs to Tensorboard"
)
def umap_predictions(**kwargs):
    """
    Plot UMAP embeddings and optionally log images to Tensorboard.

    Args:
        directory (str): Directory where embeddings to plot are stored.
        categories (tuple[str]): List of categories to color by.
        keys (tuple[str]): List of embedding keys that prefix save_paths.
        save_dir (str): Path to save UMAP outputs.
        skip_tensorboard (bool): Prevent logging UMAPs to Tensorboard.
    """
    generate_umap(**kwargs)


if __name__ == "__main__":
    umap_predictions()
