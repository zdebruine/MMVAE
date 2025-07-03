import os
import glob
import torch
import argparse as ap

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics.functional import pairwise_euclidean_distance

from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cross_generation import CrossGenerator
from cmmvae.runners.cell_plot import plot_heatmap

def convert_to_tensor(data: sp.csr_matrix, return_dense: bool = True):
    tensor = torch.sparse_csr_tensor(
        crow_indices=data.indptr,
        col_indices=data.indices,
        values=data.data,
        size=data.shape,
        dtype=torch.float32,
    )
    if return_dense:
        tensor = tensor.to_dense()
    return tensor

def pairwise_pearson_log(X, Y):

    Xc = X - X.mean(dim=1, keepdim=True)
    Yc = Y - Y.mean(dim=1, keepdim=True)

    cov = (Xc @ Yc.T)
    var_x = (Xc.pow(2).sum(dim=1))
    var_y = (Yc.pow(2).sum(dim=1))

    log_cov = torch.log(cov.abs() + 1e-6)
    log_var_x = torch.log(var_x + 1e-6) * 0.5
    log_var_y = torch.log(var_y + 1e-6) * 0.5

    L = log_cov - log_var_x.unsqueeze(1) - log_var_y

    return torch.exp(L) * torch.sign(cov)

DATA_PATH = "/mnt/projects/debruinz_project/july2024_census_data/cells/"

def main(directory: str):
    
    model = CrossGenerator(directory)

    output_dir = os.path.join(directory, "cell_correlations")
    os.makedirs(output_dir, exist_ok=True)

    human_npzs = glob.glob(os.path.join(DATA_PATH, "human_*.npz"))
    human_pkls = glob.glob(os.path.join(DATA_PATH, "human_*.pkl"))
    mouse_npzs = glob.glob(os.path.join(DATA_PATH, "mouse_*.npz"))
    mouse_pkls = glob.glob(os.path.join(DATA_PATH, "mouse_*.pkl"))

    human_npzs.sort()
    human_pkls.sort()
    mouse_npzs.sort()
    mouse_pkls.sort()

    print("Human NPZs: ", human_npzs, flush=True)
    print("Human PKLs: ", human_pkls, flush=True)
    print("Mouse NPZs: ", mouse_npzs, flush=True)
    print("Mouse PKLs: ", mouse_pkls, flush=True)

    human = pd.read_csv(os.path.join(DATA_PATH, "human_counts.csv"))
    mouse = pd.read_csv(os.path.join(DATA_PATH, "mouse_counts.csv"))

    columns = human['cell_type']
    index = mouse['cell_type']

    to_human_correlations = pd.DataFrame(index=index, columns=columns)
    to_mouse_correlations = pd.DataFrame(index=index, columns=columns)

    to_human_distances = pd.DataFrame(index=index, columns=columns)
    to_mouse_distances = pd.DataFrame(index=index, columns=columns)

    for human_cell_type, human_npz, human_pkl in zip(columns, human_npzs, human_pkls):

        print(f"Human Cell Type: {human_cell_type}", flush=True)

        human_metadata = pd.read_pickle(human_pkl)
        human_metadata["species"] = RK.HUMAN

        human_sample = human_metadata.sample(n=min(len(human_metadata), 50000), random_state=42)
        
        human_data = sp.load_npz(human_npz).astype(np.float32)
        human_batch_data = convert_to_tensor(human_data[human_sample.index.tolist(), :])
        human_batch_metadata = human_sample.reset_index(drop=True)

        human_to_human, human_z = model.get_cis_outputs(human_batch_data, human_batch_metadata, expert_id=RK.HUMAN)
        human_to_mouse_df = human_batch_metadata.copy(deep=True)
        human_to_mouse_df["species"] = RK.MOUSE
        human_to_mouse = model._get_xhat(human_z, human_to_mouse_df, expert_id=RK.MOUSE, species=RK.HUMAN)

        for mouse_cell_type, mouse_npz, mouse_pkl in zip(columns, mouse_npzs, mouse_pkls):

            print(f"Mouse Cell Type: {mouse_cell_type}", flush=True)

            mouse_metadata = pd.read_pickle(mouse_pkl)
            mouse_metadata["species"] = RK.MOUSE

            mouse_sample = mouse_metadata.sample(n=min(len(mouse_metadata), 50000), random_state=42)
            
            mouse_data = sp.load_npz(mouse_npz).astype(np.float32)
            mouse_batch_data = convert_to_tensor(mouse_data[mouse_sample.index.tolist(), :])
            mouse_batch_metadata = mouse_sample.reset_index(drop=True)

            mouse_to_mouse, mouse_z = model.get_cis_outputs(mouse_batch_data, mouse_batch_metadata, expert_id=RK.MOUSE)
            mouse_to_human_df = mouse_batch_metadata.copy(deep=True)
            mouse_to_human_df["species"] = RK.HUMAN
            mouse_to_human = model._get_xhat(mouse_z, mouse_to_human_df, expert_id=RK.HUMAN, species=RK.MOUSE)

            to_human_correlation = pairwise_pearson_log(human_to_human, mouse_to_human).mean().item()
            to_mouse_correlation = pairwise_pearson_log(mouse_to_mouse, human_to_mouse).mean().item()

            to_human_correlations.loc[mouse_cell_type, human_cell_type] = round(to_human_correlation, 2)
            to_mouse_correlations.loc[mouse_cell_type, human_cell_type] = round(to_mouse_correlation, 2)

            to_human_distance = pairwise_euclidean_distance(human_to_human, mouse_to_human).mean().item()
            to_mouse_distance = pairwise_euclidean_distance(mouse_to_mouse, human_to_mouse).mean().item()

            to_human_distances.loc[mouse_cell_type, human_cell_type] = round(to_human_distance, 2)
            to_mouse_distances.loc[mouse_cell_type, human_cell_type] = round(to_mouse_distance, 2)

    to_human_correlations['Row Mean'] = to_human_correlations[columns].mean(axis=1).astype('float64').round(2)
    to_human_correlations['Row STD'] = to_human_correlations[columns].std(axis=1).astype('float64').round(2)

    cmean = to_human_correlations[columns].mean().astype('float64').round(2)
    cstd = to_human_correlations[columns].std().astype('float64').round(2)

    to_human_correlations.loc['Column Mean', columns] = cmean
    to_human_correlations.loc['Column STD', columns] = cstd

    to_mouse_correlations['Row Mean'] = to_mouse_correlations[columns].mean(axis=1).astype('float64').round(2)
    to_mouse_correlations['Row STD'] = to_mouse_correlations[columns].std(axis=1).astype('float64').round(2)

    cmean = to_mouse_correlations[columns].mean().astype('float64').round(2)
    cstd = to_mouse_correlations[columns].std().astype('float64').round(2)

    to_mouse_correlations.loc['Column Mean', columns] = cmean
    to_mouse_correlations.loc['Column STD', columns] = cstd

    to_human_correlations.to_csv(os.path.join(output_dir, "to_human_correlations.csv"))
    to_mouse_correlations.to_csv(os.path.join(output_dir, "to_mouse_correlations.csv"))

    to_human_distances['Row Mean'] = to_human_distances.mean(axis=1).astype('float64').round(2)
    to_human_distances['Row STD'] = to_human_distances.std(axis=1).astype('float64').round(2)

    cmean = to_human_distances[columns].mean().astype('float64').round(2)
    cstd = to_human_distances[columns].std().astype('float64').round(2)

    to_human_distances.loc['Column Mean', columns] = cmean
    to_human_distances.loc['Column STD', columns] = cstd

    to_mouse_distances['Row Mean'] = to_mouse_distances.mean(axis=1).astype('float64').round(2)
    to_mouse_distances['Row STD'] = to_mouse_distances.std(axis=1).astype('float64').round(2)

    cmean = to_mouse_distances[columns].mean().astype('float64').round(2)
    cstd = to_mouse_distances[columns].std().astype('float64').round(2)

    to_mouse_distances.loc['Column Mean', columns] = cmean
    to_mouse_distances.loc['Column STD', columns] = cstd

    to_human_distances.to_csv(os.path.join(output_dir, "to_human_distances.csv"))
    to_mouse_distances.to_csv(os.path.join(output_dir, "to_mouse_distances.csv"))

    # List the labels you want to drop
    drop_idx = ["Row Mean", "Row STD", "Column Mean", "Column STD"]

    # For correlations
    heatmap_corr = to_human_correlations.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    # Similarly for distances (if you also want a heatmap of those)
    heatmap_dist = to_human_distances.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    plot_heatmap(heatmap_corr.astype('float64'),
                title="Pairwise Pearson Correlations (Human)",
                out_path=os.path.join(output_dir, "to_human_correlations.png"),
                cbar_label="Correlation")

    heatmap_corr = heatmap_corr ** 2

    plot_heatmap(heatmap_corr.astype('float64'),
                title="Pairwise R^2 Correlations (Human)",
                out_path=os.path.join(output_dir, "to_human_r2_correlations.png"),
                # annot_kws={"size": 12, "weight": "bold", "color": "white"},
                cbar_label="Correlation")

    values = heatmap_dist.to_numpy().flatten()
    values.sort()

    max_val = (values[-1] // 5) * 5 + 5
    all_ticks = np.arange(0, max_val + 1, 5)

    ticks = [
        0,
        all_ticks[len(all_ticks) // 4],
        all_ticks[len(all_ticks) // 2],
        all_ticks[len(all_ticks) // 4 * 3],
        max_val
    ]

    plot_heatmap(heatmap_dist.astype('float64'),
                title="Pairwise Euclidean Distances (Human)",
                out_path=os.path.join(output_dir, "to_human_distances.png"),
                annot_kws={"size": 12, "weight": "bold", "color": "white"},
                cmap="rocket_r",
                cbar_ticks=ticks,
                cbar_label="Distance")

    # For correlations
    heatmap_corr = to_mouse_correlations.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    # Similarly for distances (if you also want a heatmap of those)
    heatmap_dist = to_mouse_distances.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    plot_heatmap(heatmap_corr.astype('float64'),
                title="Pairwise Pearson Correlations (Mouse)",
                out_path=os.path.join(output_dir, "to_mouse_correlations.png"),
                cbar_label="Correlation")

    heatmap_corr = heatmap_corr ** 2

    plot_heatmap(heatmap_corr.astype('float64'),
                title="Pairwise R^2 Correlations (Mouse)",
                out_path=os.path.join(output_dir, "to_mouse_r2_correlations.png"),
                cbar_label="Correlation")

    values = heatmap_dist.to_numpy().flatten()
    values.sort()

    max_val = (values[-1] // 5) * 5 + 5
    all_ticks = np.arange(0, max_val + 1, 5)

    ticks = [
        0,
        all_ticks[len(all_ticks) // 4],
        all_ticks[len(all_ticks) // 2],
        all_ticks[len(all_ticks) // 4 * 3],
        max_val
    ]

    plot_heatmap(heatmap_dist.astype('float64'),
                title="Pairwise Euclidean Distances (Mouse)",
                out_path=os.path.join(output_dir, "to_mouse_distances.png"),
                annot_kws={"size": 12, "weight": "bold", "color": "white"},
                cmap="rocket_r",
                cbar_ticks=ticks,
                cbar_label="Distance")
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    main(args.directory)