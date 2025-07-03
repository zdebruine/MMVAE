import os
import torch
import math
import argparse as ap

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics.functional import pairwise_euclidean_distance

from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cross_generation import CrossGenerator

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

def plot_heatmap(df, title, out_path,
                 figsize=(16, 14),
                 cmap="rocket",             
                 annot_fmt=".2f",
                 annot_kws={"size": 6, "weight": "bold", "color": "white"},
                 cbar_label="",
                 cbar_ticks=[0, 0.25, 0.5, 0.75, 1],
                 square=True,
                 font_scale=1.2):
    """
    Plots and saves a seaborn heatmap with nicer defaults.
    """
    plt.figure(figsize=figsize)

    sns.set_context(font_scale=font_scale)
    sns.set_style("whitegrid")

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=annot_fmt,
        annot_kws=annot_kws,
        cmap=cmap,
        linewidths=0.5,
        vmin=cbar_ticks[0],
        vmax=cbar_ticks[-1],
        cbar_kws={"label": cbar_label, "ticks":cbar_ticks, "shrink": 0.8},
        square=square,
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, size=18, weight="bold")
    cbar.ax.yaxis.set_label_position('left')

    # rotate labels for readability
    plt.xticks(rotation=45, ha="left")
    plt.yticks(rotation=0)
    ax.set_title(title, pad=32, fontsize=font_scale * 20, weight="semibold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

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

DATA_PATH = "/mnt/projects/debruinz_project/july2024_census_data/jin/"

def main(directory: str):
    
    model = CrossGenerator(directory)

    output_dir = os.path.join(directory, "jin_cell_correlations")
    os.makedirs(output_dir, exist_ok=True)

    human_npz = os.path.join(DATA_PATH, "human_counts_subset.npz")
    human_pkl = os.path.join(DATA_PATH, "human_metadata_subset.pkl")
    mouse_npz = os.path.join(DATA_PATH, "mouse_counts_subset.npz")
    mouse_pkl = os.path.join(DATA_PATH, "mouse_metadata_subset.pkl")

    human_data = convert_to_tensor(sp.load_npz(human_npz).astype(np.float32))
    human_metadata = pd.read_pickle(human_pkl)
    human_metadata["species"] = RK.HUMAN

    mouse_data = convert_to_tensor(sp.load_npz(mouse_npz).astype(np.float32))
    mouse_metadata = pd.read_pickle(mouse_pkl)
    mouse_metadata["species"] = RK.MOUSE

    human = human_metadata.groupby("cell_type", observed=True, sort=False)
    mouse = mouse_metadata.groupby("cell_type", observed=True, sort=False)

    columns = human.groups.keys()
    index = mouse.groups.keys()

    to_human_correlations = pd.DataFrame(index=index, columns=columns)
    to_mouse_correlations = pd.DataFrame(index=index, columns=columns)

    to_human_distances = pd.DataFrame(index=index, columns=columns)
    to_mouse_distances = pd.DataFrame(index=index, columns=columns)

    for human_cell_type, human_df in human:

        human_sample = human_df.sample(n=min(len(human_df), 50000), random_state=42)
        human_batch_data = human_data[human_sample.index.tolist(), :]
        human_batch_metadata = human_metadata.loc[human_sample.index.tolist(), :]

        human_to_human, human_z = model.get_cis_outputs(human_batch_data, human_batch_metadata, expert_id=RK.HUMAN)
        human_to_mouse_df = human_batch_metadata.copy(deep=True)
        human_to_mouse_df["species"] = RK.MOUSE
        human_to_mouse = model._get_xhat(human_z, human_to_mouse_df, expert_id=RK.MOUSE, species=RK.HUMAN)

        print(f"human_cell_type: {human_cell_type}")
        
        # stds = human_to_human.std(dim=1)        # shape: (N,)
        # num_zero = (stds == 0).sum().item()
        # print(f"{num_zero}/{stds.size(0):.2f}% human-human samples have zero variance")

        # stds = human_to_mouse.std(dim=1)        # shape: (N,)
        # num_zero = (stds == 0).sum().item()
        # print(f"{num_zero}/{stds.size(0):.2f}% human-mouse samples have zero variance")

        for mouse_cell_type, mouse_df in mouse:

            mouse_sample = mouse_df.sample(n=min(len(mouse_df), 50000), random_state=42)
            mouse_batch_data = mouse_data[mouse_sample.index.tolist(), :]
            mouse_batch_metadata = mouse_metadata.loc[mouse_sample.index.tolist(), :]

            mouse_to_mouse, mouse_z = model.get_cis_outputs(mouse_batch_data, mouse_batch_metadata, expert_id=RK.MOUSE)
            mouse_to_human_df = mouse_batch_metadata.copy(deep=True)
            mouse_to_human_df["species"] = RK.HUMAN
            mouse_to_human = model._get_xhat(mouse_z, mouse_to_human_df, expert_id=RK.HUMAN, species=RK.MOUSE)

            print(f"mouse_cell_type: {mouse_cell_type}")

            # stds = mouse_to_mouse.std(dim=1)        # shape: (N,)
            # num_zero = (stds == 0).sum().item()
            # print(f"{num_zero}/{stds.size(0):.2f}% mouse-mouse samples have zero variance")

            # stds = mouse_to_human.std(dim=1)        # shape: (N,)
            # num_zero = (stds == 0).sum().item()
            # print(f"{num_zero}/{stds.size(0):.2f}% mouse-human samples have zero variance")

            # sample_size = min(
            #     math.floor(len(human_sample) * 0.75),
            #     math.floor(len(mouse_sample) * 0.75)
            # )

            # human_idx = np.random.choice(human_to_human.shape[0], size=sample_size, replace=False)
            # mouse_idx = np.random.choice(mouse_to_mouse.shape[0], size=sample_size, replace=False)

            # to_human_correlation = pearson_corrcoef(human_to_human[human_idx, :], mouse_to_human[mouse_idx, :]).mean().item()
            # to_mouse_correlation = pearson_corrcoef(mouse_to_mouse[mouse_idx, :], human_to_mouse[human_idx, :]).mean().item()

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

    # # to_human_correlations.to_pickle(f"/mnt/projects/debruinz_project/tony_boos/multispecies_study/{model_name}_to_human_correlations.pkl")
    to_human_correlations.to_csv(os.path.join(output_dir, "to_human_correlations.csv"))

    # # to_mouse_correlations.to_pickle(f"/mnt/projects/debruinz_project/tony_boos/multispecies_study/{model_name}_to_mouse_correlations.pkl")
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

    # to_human_distances.to_pickle(f"/mnt/projects/debruinz_project/tony_boos/multispecies_study/{model_name}_to_human_distances.pkl")
    to_human_distances.to_csv(os.path.join(output_dir, "to_human_distances.csv"))

    # to_mouse_distances.to_pickle(f"/mnt/projects/debruinz_project/tony_boos/multispecies_study/{model_name}_to_mouse_distances.pkl")
    to_mouse_distances.to_csv(os.path.join(output_dir, "to_mouse_distances.csv"))

    # List the labels you want to drop
    drop_idx = ["Row Mean", "Row STD", "Column Mean", "Column STD"]

    # For correlations
    heatmap_corr = to_human_correlations.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    # Similarly for distances (if you also want a heatmap of those)
    heatmap_dist = to_human_distances.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")
    
    shared_corr = sorted(set(heatmap_corr.index).intersection(heatmap_corr.columns))
    shared_dist = sorted(set(heatmap_dist.index).intersection(heatmap_dist.columns))

    rows_only_corr = [x for x in heatmap_corr.index   if x not in shared_corr]
    cols_only_corr = [x for x in heatmap_corr.columns if x not in shared_corr]

    rows_only_dist = [x for x in heatmap_dist.index   if x not in shared_dist]
    cols_only_dist = [x for x in heatmap_dist.columns if x not in shared_dist]

    new_index   = shared_corr + rows_only_corr
    new_columns = shared_corr + cols_only_corr

    heatmap_corr = heatmap_corr.reindex(index=new_index, columns=new_columns)

    new_index   = shared_dist + rows_only_dist
    new_columns = shared_dist + cols_only_dist

    heatmap_dist = heatmap_dist.reindex(index=new_index, columns=new_columns)


    plot_heatmap(heatmap_corr,
                title="Pairwise Pearson Correlations (Human)",
                out_path=os.path.join(output_dir, "to_human_correlations.png"),
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

    plot_heatmap(heatmap_dist,
                title="Pairwise Euclidean Distances (Human)",
                out_path=os.path.join(output_dir, "to_human_distances.png"),
                cmap="rocket_r",
                cbar_ticks=ticks,
                cbar_label="Distance")

    # For correlations
    heatmap_corr = to_mouse_correlations.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    # Similarly for distances (if you also want a heatmap of those)
    heatmap_dist = to_mouse_distances.drop(index=drop_idx, errors="ignore") \
                                    .drop(columns=drop_idx, errors="ignore")

    shared_corr = sorted(set(heatmap_corr.index).intersection(heatmap_corr.columns))
    shared_dist = sorted(set(heatmap_dist.index).intersection(heatmap_dist.columns))

    rows_only_corr = [x for x in heatmap_corr.index   if x not in shared_corr]
    cols_only_corr = [x for x in heatmap_corr.columns if x not in shared_corr]

    rows_only_dist = [x for x in heatmap_dist.index   if x not in shared_dist]
    cols_only_dist = [x for x in heatmap_dist.columns if x not in shared_dist]

    new_index   = shared_corr + rows_only_corr
    new_columns = shared_corr + cols_only_corr

    heatmap_corr = heatmap_corr.reindex(index=new_index, columns=new_columns)

    new_index   = shared_dist + rows_only_dist
    new_columns = shared_dist + cols_only_dist

    heatmap_dist = heatmap_dist.reindex(index=new_index, columns=new_columns)

    plot_heatmap(heatmap_corr,
                title="Pairwise Pearson Correlations (Mouse)",
                out_path=os.path.join(output_dir, "to_mouse_correlations.png"),
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

    plot_heatmap(heatmap_dist,
                title="Pairwise Euclidean Distances (Mouse)",
                out_path=os.path.join(output_dir, "to_mouse_distances.png"),
                cmap="rocket_r",
                cbar_ticks=ticks,
                cbar_label="Distance")
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    main(args.directory)