"""
Get R^2 correlations between cis and cross species generations
"""
import os
import glob
import torch


import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional

from torch import Tensor
from torch.nn.functional import mse_loss
from torchmetrics.functional import  pairwise_euclidean_distance
from cmmvae.runners.cross_generation import CrossGenerator

FILE_PATTERN = 'human_filtered_'
EPSILON = 1e-4

def pairwise_mse(
    x: Tensor,
    y: Optional[Tensor] = None,
) -> Tensor:

    if y is None:
        y = x

    x = x.unsqueeze(1).expand(-1, x.shape[0], -1)
    y = y.unsqueeze(0).expand(y.shape[0], -1, -1)
    mse = mse_loss(x, y, reduction="none")
    summed = mse.sum(dim=-1)

    return summed

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

PATH = "/mnt/projects/debruinz_project/disease_study/datasets"
DISEASES = {"covid": "COVID-19", "crohn": "Crohn disease", "lung_adenocarcinoma": "lung adenocarcinoma"}
CORRELATION_FUNCTIONS = {"MSE": pairwise_mse, "Euclidean": pairwise_euclidean_distance}#, "Cosine": pairwise_cosine_similarity}

for model_type in ["full", "filtered"]:
    model = CrossGenerator(f"/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/{model_type}")
    for disease, col_name in DISEASES.items():
        npz = glob.glob(os.path.join(PATH, f"full_{disease}*.npz"))
        pkl = glob.glob(os.path.join(PATH, f"full_{disease}*.pkl"))
        npz.sort()
        pkl.sort()

        mse = pd.DataFrame(
            columns=[
                "dataset_id",
                "cell_and_tissue_types",
                "disease_mse",
                "normal_mse",
            ]
        )

        for mat, df in zip(npz, pkl):
            data = sp.load_npz(mat)
            metadata = pd.read_pickle(df)

            grouped = metadata.groupby(["cell_type", "tissue_general"], observed=True)

            for key, group in grouped:
                indices = group.index.tolist()
                df = metadata.loc[indices]
                disease_idx = df[df['disease'] == col_name].index.tolist()
                normal_idx = df[df['disease'] == 'normal'].index.tolist()

                if len(normal_idx) < 50 or len(disease_idx) < 50:
                    continue

                true_disease_data = convert_to_tensor(data[disease_idx, :])
                true_normal_data = convert_to_tensor(data[normal_idx, :])

                true_disease_metadata = metadata.loc[disease_idx]
                true_normal_metadata = metadata.loc[normal_idx]

                disease_to_disease = model.get_cis_outputs(true_disease_data, true_disease_metadata, return_z=False)

                normal_to_normal = model.get_cis_outputs(true_normal_data, true_normal_metadata, return_z=False)

                disease_mse = mse_loss(disease_to_disease, true_disease_data).item()
                normal_mse = mse_loss(normal_to_normal, true_normal_data).item()

                group_mse = pd.DataFrame(
                    {
                        "dataset_id": [metadata['dataset_id'].loc[0]],
                        "cell_and_tissue_types": [" ".join(key)],
                        "disease_mse": [round(disease_mse, 3)],
                        "normal_mse": [round(normal_mse, 3)],
                    }
                )
                mse = pd.concat([mse, group_mse])
        
        mse.to_csv(f"/mnt/projects/debruinz_project/tony_boos/disease_study/mse/{model_type}_{disease}_mse.csv", index=False)



        # width = max(8, min(2 * correlations["dataset_id"].nunique(), 15))  # Keep within a reasonable range
        # plt.figure(figsize=(width, 8))
        # sns.violinplot(x="dataset_id", y="true_and_cis_over_true_and_cross", data=correlations, inner=None, color=".8")
        # ax = sns.swarmplot(x="dataset_id", y="true_and_cis_over_true_and_cross", data=correlations, color="k")
        # plt.title(f"{disease.title()} Cross-Generation Ratio By Dataset For Cell and Tissue Type Contexts")

        # ticks = np.arange(0.94, 1.02, 0.02)

        # plt.xlabel("Dataset-ID")
        # plt.xticks(rotation=15)

        # plt.ylabel("Euclidean Distance Ratio")
        # plt.yticks(ticks)

        # plt.tight_layout()
        # plt.savefig(f"/mnt/projects/debruinz_project/tony_boos/disease_study/correlations/{model_type}_{disease}_correlations.png")
        # plt.clf()
        # plt.close()
