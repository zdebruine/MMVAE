import os
import glob
# import torch
# import argparse as ap

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
# import matplotlib.pyplot as plt

# from scipy.stats import linregress
# from anndata import AnnData
# from torchmetrics.functional import pairwise_euclidean_distance

# from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cross_generation import CrossGenerator

from cmmvae.runners.save_h5 import save_as_h5


PATH = "/mnt/projects/debruinz_project/disease_study/datasets/"
DISEASES = [
    "COVID-19",
    "Crohn disease",
    "lung adenocarcinoma",
]

def main(directory: str) -> None:

    model_name = os.path.basename(directory)

    files = [
        f"{model_name}_covid_9dbab10c-118d-496b-966a-67f1763a6b7d",
        f"{model_name}_crohn_fe4b89d5-461e-440c-a5a8-621b37b122c0",
        f"{model_name}_lung_adenocarcinoma_576f193c-75d0-4a11-bd25-8676587e6dc2"
    ]

    genes = pd.read_pickle("/mnt/projects/debruinz_project/july2024_census_data/human_gene_metadata.pkl")

    model = CrossGenerator(directory)

    for disease_name, filename in zip(DISEASES, files):

        print(filename)

        data = model._convert_to_tensor(sp.load_npz(os.path.join(PATH, f"{filename}.npz")))
        metadata = pd.read_pickle(os.path.join(PATH, f"{filename}.pkl"))

        disease_idx = metadata[metadata["disease"] == disease_name].index
        normal_idx = metadata[metadata["disease"] == "normal"].index

        true_disease_data = data[disease_idx]
        true_disease_metadata = metadata.loc[disease_idx]

        true_normal_data = data[normal_idx]
        true_normal_metadata = metadata.loc[normal_idx]

        # Get Cis/Cross outputs
        disease_to_disease, disease_z = model.get_cis_outputs(true_disease_data, true_disease_metadata)
        disease_to_normal = model.get_cross_outputs(disease_z, true_disease_metadata, true_normal_metadata, mod_tags=("disease"))

        normal_to_normal, normal_z = model.get_cis_outputs(true_normal_data, true_normal_metadata)
        normal_to_disease = model.get_cross_outputs(normal_z, true_normal_metadata, true_disease_metadata, mod_tags=("disease"))

        disease_to_disease = sp.csc_matrix(disease_to_disease.cpu().numpy())
        disease_to_normal = sp.csc_matrix(disease_to_normal.cpu().numpy())

        normal_to_normal = sp.csc_matrix(normal_to_normal.cpu().numpy())
        normal_to_disease = sp.csc_matrix(normal_to_disease.cpu().numpy())

        save_as_h5(
            disease_to_disease,
            true_disease_metadata,
            genes,
            os.path.join(directory, "expressions"),
            f"{filename}_disease_to_disease"
        )
        save_as_h5(
            disease_to_normal,
            true_disease_metadata,
            genes,
            os.path.join(directory, "expressions"),
            f"{filename}_disease_to_normal"
        )
        save_as_h5(
            normal_to_normal,
            true_normal_metadata,
            genes,
            os.path.join(directory, "expressions"),
            f"{filename}_normal_to_normal"
        )
        save_as_h5(
            normal_to_disease,
            true_normal_metadata,
            genes,
            os.path.join(directory, "expressions"),
            f"{filename}_normal_to_disease"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the model directory.",
    )

    args = parser.parse_args()

    main(args.directory)