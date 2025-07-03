"""
Get R^2 correlations between cis and cross species generations
"""
import os
import glob
import torch
import gc

import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import Optional

from torch import Tensor
from torch.nn.functional import mse_loss
from torchmetrics.functional import pairwise_euclidean_distance
from cmmvae.runners.cross_generation import CrossGenerator

PATH = "/mnt/projects/debruinz_project/disease_study/datasets"
# DISEASES = {"lung_adenocarcinoma": "lung adenocarcinoma", "crohn": "Crohn disease", "covid": "COVID-19"}
DISEASES = {"covid": "COVID-19"}

SELECTED = {disease: pd.read_csv(os.path.join('/mnt/projects/debruinz_project/tony_boos/disease_study', f'{disease}_selected_cells.csv'))['cell_type'].values.tolist() for disease in DISEASES.keys()}


def convert_to_tensor(data: sp.csr_matrix, return_dense: bool = True):
    print("Converting to tensor", flush=True)
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

def get_correlations(
    true_disease_data,
    true_disease_metadata,
    true_normal_data,
    true_normal_metadata,
    dataset_id, 
    cell_type,
    tissue_type,
    withheld,
):
    disease_to_disease, disease_z = model.get_cis_outputs(true_disease_data, true_disease_metadata)
    disease_to_normal = model.get_cross_outputs(disease_z, true_disease_metadata, true_normal_metadata, mod_tags=("disease"))

    normal_to_normal, normal_z = model.get_cis_outputs(true_normal_data, true_normal_metadata)
    normal_to_disease = model.get_cross_outputs(normal_z, true_normal_metadata, true_disease_metadata, mod_tags=("disease"))

    disease_mse = mse_loss(disease_to_disease, true_disease_data).item()
    normal_mse = mse_loss(normal_to_normal, true_normal_data).item()

    true_disease_and_true_normal = pairwise_euclidean_distance(true_disease_data, true_normal_data).mean().item()

    true_disease_and_disease_to_disease = pairwise_euclidean_distance(true_disease_data, disease_to_disease).mean().item()
    true_disease_and_normal_to_disease = pairwise_euclidean_distance(true_disease_data, normal_to_disease).mean().item()

    true_normal_and_normal_to_normal = pairwise_euclidean_distance(true_normal_data, normal_to_normal).mean().item()
    true_normal_and_disease_to_normal = pairwise_euclidean_distance(true_normal_data, disease_to_normal).mean().item()

    disease_to_disease_and_normal_to_normal = pairwise_euclidean_distance(disease_to_disease, normal_to_normal).mean().item()
    disease_to_normal_and_normal_to_disease = pairwise_euclidean_distance(disease_to_normal, normal_to_disease).mean().item()

    disease_to_disease_and_disease_to_disease = pairwise_euclidean_distance(disease_to_disease).mean().item()
    normal_to_disease_and_normal_to_disease = pairwise_euclidean_distance(normal_to_disease).mean().item()
    disease_to_disease_and_normal_to_disease = pairwise_euclidean_distance(disease_to_disease, normal_to_disease).mean().item()

    normal_to_normal_and_normal_to_normal = pairwise_euclidean_distance(normal_to_normal).mean().item()
    disease_to_normal_and_disease_to_normal = pairwise_euclidean_distance(disease_to_normal).mean().item()
    normal_to_normal_and_disease_to_normal = pairwise_euclidean_distance(normal_to_normal, disease_to_normal).mean().item()
    

    true_over_true_and_cis = true_disease_and_true_normal / (2 * (true_disease_and_disease_to_disease + true_normal_and_normal_to_normal))
    true_over_true_and_cross = true_disease_and_true_normal / (2 * (true_disease_and_normal_to_disease + true_normal_and_disease_to_normal))

    cis_over_cis_and_cis = disease_to_disease_and_normal_to_normal / (2 * (disease_to_disease_and_disease_to_disease + normal_to_normal_and_normal_to_normal))
    cross_over_cross_and_cross = disease_to_normal_and_normal_to_disease / (2 * (disease_to_disease_and_normal_to_disease + normal_to_normal_and_disease_to_normal))

    cis_and_cis_over_cross_and_cross = (
        (disease_to_disease_and_disease_to_disease + normal_to_normal_and_normal_to_normal) 
        / (disease_to_disease_and_normal_to_disease + normal_to_normal_and_disease_to_normal)
    )

    return {
        "dataset_id": dataset_id,
        "tissue_type": tissue_type,
        "cell_type": cell_type,
        "withheld": withheld,
        "disease_mse": round(disease_mse, 3),
        "normal_mse": round(normal_mse, 3),
        "true_disease_and_true_normal": round(true_disease_and_true_normal, 3),
        "true_disease_and_disease_to_disease": round(true_disease_and_disease_to_disease, 3),
        "true_disease_and_normal_to_disease": round(true_disease_and_normal_to_disease, 3),
        "true_normal_and_normal_to_normal": round(true_normal_and_normal_to_normal, 3),
        "true_normal_and_disease_to_normal": round(true_normal_and_disease_to_normal, 3),
        "disease_to_disease_and_normal_to_normal": round(disease_to_disease_and_normal_to_normal, 3),
        "disease_to_normal_and_normal_to_disease": round(disease_to_normal_and_normal_to_disease, 3),
        "disease_to_disease_and_disease_to_disease": round(disease_to_disease_and_disease_to_disease, 3),
        "normal_to_disease_and_normal_to_disease": round(normal_to_disease_and_normal_to_disease, 3),
        "disease_to_disease_and_normal_to_disease": round(disease_to_disease_and_normal_to_disease, 3),
        "normal_to_normal_and_normal_to_normal": round(normal_to_normal_and_normal_to_normal, 3),
        "disease_to_normal_and_disease_to_normal": round(disease_to_normal_and_disease_to_normal, 3),
        "normal_to_normal_and_disease_to_normal": round(normal_to_normal_and_disease_to_normal, 3),
        "true_over_true_and_cis": round(true_over_true_and_cis, 3),
        "true_over_true_and_cross": round(true_over_true_and_cross, 3),
        "cis_over_cis_and_cis": round(cis_over_cis_and_cis, 3), 
        "cross_over_cross_and_cross": round(cross_over_cross_and_cross, 3),
        "cis_and_cis_over_cross_and_cross": round(cis_and_cis_over_cross_and_cross, 3),
    }

# for model_type in ["full", "filtered"]:
model = CrossGenerator(f"/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/filtered")
for disease, col_name in DISEASES.items():
    npz = sorted(glob.glob(os.path.join(PATH, f"withheld_{disease}_context*.npz")))
    pkl = sorted(glob.glob(os.path.join(PATH, f"withheld_{disease}_context*.pkl")))

    correlations = []

    for mat, df in zip(npz, pkl):
        print(df, flush=True)
        data = convert_to_tensor(sp.load_npz(mat).astype(np.float32))
        metadata = pd.read_pickle(df)

        if len(metadata) > 100000:
            sample = np.random.choice(len(metadata), 100000, replace=False)
            data = data[sample, :]
            metadata = metadata.loc[sample].reset_index(drop=True)

        # grouped = metadata.groupby(["cell_type", "tissue_general"], observed=True, sort=False)

        # for key, group in grouped:

        disease_idx = metadata[metadata['disease'] == col_name].index.tolist()
        normal_idx = metadata[metadata['disease'] == 'normal'].index.tolist()

        if len(normal_idx) < 50 or len(disease_idx) < 50:
            continue

        correlations.append(
            get_correlations(
                true_disease_data = data[disease_idx, :],
                true_disease_metadata = metadata.loc[disease_idx],
                true_normal_data = data[normal_idx, :],
                true_normal_metadata = metadata.loc[normal_idx],
                dataset_id = metadata['dataset_id'].loc[0],
                cell_type = metadata['cell_type'].loc[0],
                tissue_type = metadata['tissue_general'].loc[0],
                withheld = metadata['cell_type'].loc[0] in SELECTED[disease],
            )
        )
        # correlations = pd.concat([correlations, group_correlations])

    correlations_df = pd.DataFrame(correlations)
    correlations_df.to_csv(f"/mnt/projects/debruinz_project/tony_boos/disease_study/pairwise_mse/{model_type}_{disease}_correlations.csv", index=False)



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
