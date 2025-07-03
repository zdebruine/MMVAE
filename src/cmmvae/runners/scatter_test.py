import numpy as np
import umap
import torch, sys, os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import load_npz
from cmmvae.runners.cross_generation import CrossGenerator
from cmmvae.runners.correlations import pairwise_mse
from cmmvae.constants import REGISTRY_KEYS as RK
from torch.nn.functional import mse_loss
from torchmetrics.functional import pairwise_euclidean_distance

CORRELATION_FUNCTIONS = {"MSE": pairwise_mse, "Euclidean": pairwise_euclidean_distance}
DISEASES = ["covid", "crohn", "lung_adenocarcinoma"]

full_model = CrossGenerator("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/full")
filtered_model = CrossGenerator("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/filtered")

reducer = umap.UMAP(
    n_neighbors=30,
    # metric="precomputed",
    low_memory=False,
    n_jobs=40,
    n_epochs=200,
)

# for metric, metric_function in CORRELATION_FUNCTIONS.items():
    # embedding = reducer.fit_transform(
    #     metric_function(
    #         torch.cat(
    #             [A, A_to_A, B_to_A, B, B_to_B, A_to_B]
    #         )
    #     ).cpu().numpy()
    # )
    # print(embedding.shape)

for disease in DISEASES:

    data, metadata = full_model.get_data("/mnt/projects/debruinz_project/disease_study/selected", f"selected_{disease}_validation", sample=False)
    metadata = metadata.reset_index(drop=True)

    # assert len(metadata) > 100
    grouped = metadata.groupby(["cell_type", "tissue_general"])

    for key, group in grouped:

        indices = group.index.tolist()

        if len(indices) < 100:
            continue

        full_out = full_model.get_cis_outputs(data[indices, :], metadata.loc[indices], return_z=False)
        filtered_out = filtered_model.get_cis_outputs(data[indices, :], metadata.loc[indices], return_z=False)

        embedding = reducer.fit_transform(
                torch.cat(
                    [full_out, filtered_out]
                ).cpu().numpy()
        )


        plt.figure(figsize=(14, 8))
        labels = ["Full_model", "Filtered_model"]

        # Prepare color map
        cmap = plt.get_cmap("tab10")
        color_list = [cmap(i) for i in range(len(labels))]

        # Combine embedding and metadata into a DataFrame
        df = pd.DataFrame(embedding, columns=["x", "y"])
        df["labels"] = np.repeat(labels, len(indices))

        # Shuffle the DataFrame to randomize the plotting order
        df = df.sample(frac=1).reset_index(drop=True)

        # Create a dictionary to map categories to colors
        category_to_color = {value: color_list[i] for i, value in enumerate(labels)}

        # Map colors to the entire DataFrame
        df["color"] = df["labels"].map(category_to_color)

        # Plot all points in the shuffled order
        # with specified opacity and marker size
        plt.scatter(x=df["x"], y=df["y"], c=df["color"], s=5, alpha=0.5)

        plt.title(f"{disease} UMAP projection of {key[0]} & {key[1]} on full and filtered models")

        # Custom legend with a circle for each label
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label.decode("utf-8") if isinstance(label, bytes) else label,
                markerfacecolor=cmap(i),
                markersize=10,
            )
            for i, label in enumerate(labels)
        ]

        plt.legend(
            handles=legend_handles,
            title="Output Type",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.savefig(f"/mnt/projects/debruinz_project/tony_boos/disease_study/umaps/{disease}_umap_{key[0].replace(' ', '_').replace('/', '_')}_{key[1].replace(' ', '_').replace('/', '_')}.png", bbox_inches="tight")
        plt.close()

    # plt.figure(figsize=(14, 8))
    # labels = metadata['cell_type'].value_counts().nlargest(10).index

    # # Prepare color map
    # cmap = plt.get_cmap("tab10")
    # color_list = [cmap(i) for i in range(len(labels))]

    # # Combine embedding and metadata into a DataFrame
    # df = pd.DataFrame(embedding, columns=["x", "y"])
    # df["labels"] = np.repeat(metadata['cell_type'].values, 2)

    # # Filter to include only the largest categories
    # df = df[df["labels"].isin(labels)]

    # # Shuffle the DataFrame to randomize the plotting order
    # df = df.sample(frac=1).reset_index(drop=True)

    # # Create a dictionary to map categories to colors
    # category_to_color = {value: color_list[i] for i, value in enumerate(labels)}

    # # Map colors to the entire DataFrame
    # df["color"] = df["labels"].map(category_to_color)

    # # Plot all points in the shuffled order
    # # with specified opacity and marker size
    # plt.scatter(x=df["x"], y=df["y"], c=df["color"], s=1, alpha=0.5)

    # plt.title(f"UMAP projection of {disease} on cell type via UMAP distance")

    # # Custom legend with a circle for each label
    # legend_handles = [
    #     plt.Line2D(
    #         [0],
    #         [0],
    #         marker="o",
    #         color="w",
    #         label=label.decode("utf-8") if isinstance(label, bytes) else label,
    #         markerfacecolor=cmap(i),
    #         markersize=10,
    #     )
    #     for i, label in enumerate(labels)
    # ]

    # plt.legend(
    #     handles=legend_handles,
    #     title="Output Type",
    #     bbox_to_anchor=(1.05, 1),
    #     loc="upper left",
    # )

    # plt.savefig(f"/mnt/projects/debruinz_project/tony_boos/disease_study/umaps/{disease}_umap_cell_types.png", bbox_inches="tight")
    # plt.close()


# ctx_a = sys.argv[1]
# ctx_b = sys.argv[2]

# path = "/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/final_model/"
# MODEL = "baseline_relu_cond"

# cg = CrossGenerator(os.path.join(path, MODEL))

# A, A_df = cg.get_data("/mnt/projects/debruinz_project/july2024_census_data/filtered_human", ctx_a)
# B, B_df = cg.get_data("/mnt/projects/debruinz_project/july2024_census_data/filtered_human", ctx_b)

# A_to_A, z_A = cg.get_cis_outputs(A, A_df)
# B_to_B, z_B = cg.get_cis_outputs(B, B_df)

# A_to_B = cg.get_cross_outputs(z_A, A_df, B_df, RK.FILTER_CATEGORIES)
# B_to_A = cg.get_cross_outputs(z_B, B_df, A_df, RK.FILTER_CATEGORIES)

# true_mse = mse_loss(A, B, reduction='none').sum(dim=1)
# A_to_A_mse = mse_loss(A_to_A, A, reduction='none').sum(dim=1)
# A_to_B_mse = mse_loss(A_to_B, B, reduction='none').sum(dim=1)
# B_to_A_mse = mse_loss(B_to_A, A, reduction='none').sum(dim=1)
# B_to_B_mse = mse_loss(B_to_B, B, reduction='none').sum(dim=1)

# most_similar = torch.argmin(true_mse).item()
# least_similar = torch.argmax(true_mse).item()

# best_true_A = A[most_similar].cpu().numpy().flatten()
# best_true_B = B[most_similar].cpu().numpy().flatten()
# best_A_to_A = A_to_A[most_similar].cpu().numpy().flatten()
# best_A_to_B = A_to_B[most_similar].cpu().numpy().flatten()
# best_B_to_A = B_to_A[most_similar].cpu().numpy().flatten()
# best_B_to_B = B_to_B[most_similar].cpu().numpy().flatten()

# worst_true_A = A[least_similar].cpu().numpy().flatten()
# worst_true_B = B[least_similar].cpu().numpy().flatten()
# worst_A_to_A = A_to_A[least_similar].cpu().numpy().flatten()
# worst_A_to_B = A_to_B[least_similar].cpu().numpy().flatten()
# worst_B_to_A = B_to_A[least_similar].cpu().numpy().flatten()
# worst_B_to_B = B_to_B[least_similar].cpu().numpy().flatten()


# fig, axes = plt.subplots(2, 3, figsize= (18, 12))

# axes[0, 0].scatter(best_true_A, best_true_B, alpha=0.5, s=1, color='blue') 
# axes[0, 0].set_xlabel(f"Best Context {ctx_a}")
# axes[0, 0].set_ylabel(f"Best Context {ctx_b}")

# axes[0, 1].scatter(best_true_A, best_A_to_A, alpha=0.5, s=1, color='blue', label=f"{ctx_a} to {ctx_a}") 
# axes[0, 1].scatter(best_true_A, best_B_to_A, alpha=0.5, s=1, color='red', label=f"{ctx_b} to {ctx_a}") 
# axes[0, 1].set_xlabel(f"Best Context {ctx_a} True")
# axes[0, 1].set_ylabel(f"Best Context {ctx_a} Reconstructed")
# axes[0, 1].legend()

# axes[0, 2].scatter(best_true_B, best_B_to_B, alpha=0.5, s=1, color='blue', label=f"{ctx_b} to {ctx_b}") 
# axes[0, 2].scatter(best_true_B, best_A_to_B, alpha=0.5, s=1, color='red', label=f"{ctx_a} to {ctx_b}") 
# axes[0, 2].set_xlabel(f"Best Context {ctx_b} True")
# axes[0, 2].set_ylabel(f"Best Context {ctx_b} Reconstructed")
# axes[0, 2].legend()

# axes[1, 0].scatter(worst_true_A, worst_true_B, alpha=0.5, s=1, color='blue') 
# axes[1, 0].set_xlabel(f"Worst Context {ctx_a}")
# axes[1, 0].set_ylabel(f"Worst Context {ctx_b}")

# axes[1, 1].scatter(worst_true_A, worst_A_to_A, alpha=0.5, s=1, color='blue', label=f"{ctx_a} to {ctx_a}") 
# axes[1, 1].scatter(worst_true_A, worst_B_to_A, alpha=0.5, s=1, color='red', label=f"{ctx_b} to {ctx_a}") 
# axes[1, 1].set_xlabel(f"Worst Context {ctx_a} True")
# axes[1, 1].set_ylabel(f"Worst Context {ctx_a} Reconstructed")
# axes[1, 1].legend()

# axes[1, 2].scatter(worst_true_B, worst_B_to_B, alpha=0.5, s=1, color='blue', label=f"{ctx_b} to {ctx_b}") 
# axes[1, 2].scatter(worst_true_B, worst_A_to_B, alpha=0.5, s=1, color='red', label=f"{ctx_a} to {ctx_b}") 
# axes[1, 2].set_xlabel(f"Worst Context {ctx_b} True")
# axes[1, 2].set_ylabel(f"Worst Context {ctx_b} Reconstructed")
# axes[1, 2].legend()

# # fig.title("Case Study")
# plt.tight_layout()
# plt.savefig("/mnt/projects/debruinz_project/tony_boos/case_study_plot.png")