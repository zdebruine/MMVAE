import os
import glob
import torch
import argparse as ap

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib.pyplot as plt
import seaborn as sns

from torch.distributions import Normal
from torch.nn.functional import mse_loss

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

DATA_PATH = "/mnt/projects/debruinz_project/july2024_census_data/subset/"

def main(directory: str):
    
    model = CrossGenerator(directory)

    output_dir = os.path.join(directory, "distribution")
    os.makedirs(output_dir, exist_ok=True)

    human_data = convert_to_tensor(
        sp.load_npz(
            os.path.join(DATA_PATH, "human_counts_15.npz")
        )
    )
    human_metadata = pd.read_pickle(
        os.path.join(DATA_PATH, "human_metadata_15.pkl")
    )
    mouse_data = convert_to_tensor(
        sp.load_npz(
            os.path.join(DATA_PATH, "mouse_counts_15.npz")
        )
    )
    mouse_metadata = pd.read_pickle(
        os.path.join(DATA_PATH, "mouse_metadata_15.pkl")
    )

    human_metadata["species"] = RK.HUMAN
    mouse_metadata["species"] = RK.MOUSE

    # data = torch.cat([human_data, mouse_data], dim=0)
    # metadata = pd.concat([human_metadata, mouse_metadata], ignore_index=True)

    # idx = np.random.permutation(data.shape[0])

    # data = data[idx, :]
    # metadata = metadata.iloc[idx]

    human_mu, human_sigma, z = model.get_dist_params(
        human_data,
        expert_id=RK.HUMAN
    )

    human_output = model._get_xhat(
        z,
        human_metadata,
        expert_id=RK.HUMAN,
    )

    pz = Normal(torch.zeros_like(z), torch.ones_like(z))

    human_collapsed_output = model._get_xhat(
        pz.rsample(),
        human_metadata,
        expert_id=RK.HUMAN,
    )

    mouse_mu, mouse_sigma, z = model.get_dist_params(
        mouse_data,
        expert_id=RK.MOUSE
    )

    mouse_output = model._get_xhat(
        z,
        mouse_metadata,
        expert_id=RK.MOUSE,
    )

    pz = Normal(torch.zeros_like(z), torch.ones_like(z))

    mouse_collapsed_output = model._get_xhat(
        pz.rsample(),
        mouse_metadata,
        expert_id=RK.MOUSE,
    )

    multispecies_mu = torch.cat([human_mu, mouse_mu], dim=0)
    multispecies_sigma = torch.cat([human_sigma, mouse_sigma], dim=0)

    human_mu_mean = human_mu.mean(dim=0).round(decimals=2)
    human_mu_std = human_mu.std(dim=0).round(decimals=2)
    human_sigma_mean = human_sigma.mean(dim=0).round(decimals=2)
    human_sigma_std = human_sigma.std(dim=0).round(decimals=2)

    mouse_mu_mean = mouse_mu.mean(dim=0).round(decimals=2)
    mouse_mu_std = mouse_mu.std(dim=0).round(decimals=2)
    mouse_sigma_mean = mouse_sigma.mean(dim=0).round(decimals=2)
    mouse_sigma_std = mouse_sigma.std(dim=0).round(decimals=2)

    multispecies_mu_mean = multispecies_mu.mean(dim=0).round(decimals=2)
    multispecies_mu_std = multispecies_mu.std(dim=0).round(decimals=2)
    multispecies_sigma_mean = multispecies_sigma.mean(dim=0).round(decimals=2)
    multispecies_sigma_std = multispecies_sigma.std(dim=0).round(decimals=2)

    human_normal_model = round(mse_loss(human_collapsed_output, human_output).item(), 2)
    human_model_true = round(mse_loss(human_output, human_data).item(), 2)
    human_normal_true = round(mse_loss(human_collapsed_output, human_data).item(), 2)

    mouse_normal_model = round(mse_loss(mouse_collapsed_output, mouse_output).item(), 2)
    mouse_model_true = round(mse_loss(mouse_output, mouse_data).item(), 2)
    mouse_normal_true = round(mse_loss(mouse_collapsed_output, mouse_data).item(), 2)

    df = pd.DataFrame({
        "human_normal_model": [human_normal_model],
        "human_model_true": [human_model_true],
        "human_normal_true": [human_normal_true],
        "mouse_normal_model": [mouse_normal_model],
        "mouse_model_true": [mouse_model_true],
        "mouse_normal_true": [mouse_normal_true],
    })
    df.to_csv(os.path.join(output_dir, "mse.csv"), index=False)

    df = pd.DataFrame({
        "human_mu_mean": human_mu_mean.numpy(),
        "human_mu_std": human_mu_std.numpy(),
        "human_sigma_mean": human_sigma_mean.numpy(),
        "human_sigma_std": human_sigma_std.numpy(),
        "mouse_mu_mean": mouse_mu_mean.numpy(),
        "mouse_mu_std": mouse_mu_std.numpy(),
        "mouse_sigma_mean": mouse_sigma_mean.numpy(),
        "mouse_sigma_std": mouse_sigma_std.numpy(),
        "multispecies_mu_mean": multispecies_mu_mean.numpy(),
        "multispecies_mu_std": multispecies_mu_std.numpy(),
        "multispecies_sigma_mean": multispecies_sigma_mean.numpy(),
        "multispecies_sigma_std": multispecies_sigma_std.numpy(),
    })
    df.to_csv(os.path.join(output_dir, "latent_metrics.csv"), index=False)

    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Boxplot of latent-dimension metrics (mean & std) across species")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_metrics.png"))

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    main(args.directory)