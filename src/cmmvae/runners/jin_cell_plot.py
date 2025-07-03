import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    df = df.astype('float64')

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

PATH = "/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/cycle_consistency/batch_norm_cycle/jin_cell_correlations"

to_human_correlations = pd.read_csv(os.path.join(PATH, "to_human_correlations.csv"), index_col=0)
to_mouse_correlations = pd.read_csv(os.path.join(PATH, "to_mouse_correlations.csv"), index_col=0)
to_human_distances = pd.read_csv(os.path.join(PATH, "to_human_distances.csv"), index_col=0)
to_mouse_distances = pd.read_csv(os.path.join(PATH, "to_mouse_distances.csv"), index_col=0)

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

plot_heatmap(heatmap_corr.loc[shared_corr, shared_corr],
             title="Pairwise Pearson Correlations (Human)",
             out_path=os.path.join(PATH, "to_human_correlations.png"),
             annot_kws={"size": 12, "weight": "bold", "color": "white"},
             cbar_label="Correlation")

plot_heatmap(heatmap_corr,
             title="Pairwise Pearson Correlations (Human)",
             out_path=os.path.join(PATH, "to_human_correlations_full.png"),
             cbar_label="Correlation")

heatmap_corr = heatmap_corr ** 2

plot_heatmap(heatmap_corr.loc[shared_corr, shared_corr],
             title="Pairwise R^2 Correlations (Human)",
             out_path=os.path.join(PATH, "to_human_r2_correlations.png"),
             annot_kws={"size": 12, "weight": "bold", "color": "white"},
             cbar_label="Correlation")

plot_heatmap(heatmap_corr,
             title="Pairwise R^2 Correlations (Human)",
             out_path=os.path.join(PATH, "to_human_r2_correlations_full.png"),
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

plot_heatmap(heatmap_dist.loc[shared_dist, shared_dist],
             title="Pairwise Euclidean Distances (Human)",
             out_path=os.path.join(PATH, "to_human_distances.png"),
             annot_kws={"size": 12, "weight": "bold", "color": "white"},
             cmap="rocket_r",
             cbar_ticks=ticks,
             cbar_label="Distance")

plot_heatmap(heatmap_dist,
             title="Pairwise Euclidean Distances (Human)",
             out_path=os.path.join(PATH, "to_human_distances_full.png"),
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

plot_heatmap(heatmap_corr.loc[shared_corr, shared_corr],
             title="Pairwise Pearson Correlations (Mouse)",
             out_path=os.path.join(PATH, "to_mouse_correlations.png"),
             annot_kws={"size": 12, "weight": "bold", "color": "white"},
             cbar_label="Correlation")

plot_heatmap(heatmap_corr,
             title="Pairwise Pearson Correlations (Mouse)",
             out_path=os.path.join(PATH, "to_mouse_correlations_full.png"),
             cbar_label="Correlation")

heatmap_corr = heatmap_corr ** 2

plot_heatmap(heatmap_corr.loc[shared_corr, shared_corr],
             title="Pairwise R^2 Correlations (Mouse)",
             out_path=os.path.join(PATH, "to_mouse_r2_correlations.png"),
             annot_kws={"size": 12, "weight": "bold", "color": "white"},
             cbar_label="Correlation")

plot_heatmap(heatmap_corr,
             title="Pairwise R^2 Correlations (Mouse)",
             out_path=os.path.join(PATH, "to_mouse_r2_correlations_full.png"),
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

plot_heatmap(heatmap_dist.loc[shared_dist, shared_dist],
             title="Pairwise Euclidean Distances (Mouse)",
             out_path=os.path.join(PATH, "to_mouse_distances.png"),
             annot_kws={"size": 12, "weight": "bold", "color": "white"},
             cmap="rocket_r",
             cbar_ticks=ticks,
             cbar_label="Distance")

plot_heatmap(heatmap_dist,
             title="Pairwise Euclidean Distances (Mouse)",
             out_path=os.path.join(PATH, "to_mouse_distances_full.png"),
             cmap="rocket_r",
             cbar_ticks=ticks,
             cbar_label="Distance")