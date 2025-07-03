import os

import scib
import torch

import argparse as ap
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from cmmvae.runners.cross_generation import CrossGenerator
from cmmvae.constants import REGISTRY_KEYS as RK

PATH = "/mnt/projects/debruinz_project/july2024_census_data/subset/"

def main(directory: str):
    model = CrossGenerator(directory)

    human_data = model._convert_to_tensor(sp.load_npz(os.path.join(PATH, "human_counts_15.npz")))
    human_metadata = pd.read_pickle(os.path.join(PATH, "human_metadata_15.pkl"))
    human_metadata["species"] = RK.HUMAN

    mouse_data = model._convert_to_tensor(sp.load_npz(os.path.join(PATH, "mouse_counts_15.npz")))
    mouse_metadata = pd.read_pickle(os.path.join(PATH, "mouse_metadata_15.pkl"))
    mouse_metadata["species"] = RK.MOUSE

    human_z = model._get_z(human_data, RK.HUMAN)
    mouse_z = model._get_z(mouse_data, RK.MOUSE)

    latents = torch.cat([human_z, mouse_z], dim=0).cpu().numpy()
    latent_md = pd.concat([human_metadata, mouse_metadata], axis=0, ignore_index=True)

    idx = np.random.permutation(latents.shape[0])

    latents = latents[idx]
    latent_md = latent_md.iloc[idx]
    latent_md = latent_md.reset_index(drop=True)

    latent_md['cell_type'] = latent_md['cell_type'].astype('category')
    latent_md['species'] = latent_md['species'].astype('category')

    # Wrap embedding & metadata into AnnData for SCIB
    adata = ad.AnnData(X= None, obs= latent_md)
    adata.obsm['X_emb'] = latents

    results = scib.metrics.metrics(
        adata,
        adata,                    # integrated AnnData (same as input if only embeddings)
        batch_key='species',      # your batch column in adata.obs
        label_key='cell_type',    # your biological label column
        embed='X_emb',            # name of your embedding in adata.obsm
        silhouette_=True,         # compute both batch and label silhouettes
        ari_=True,                # adjusted Rand index
        nmi_=True,                # normalized mutual information
        ilisi_=True,              # integration LISI
        clisi_=True,              # cell‐type LISI
        isolated_labels_asw_=False,
        isolated_labels_f1_=False,
        graph_conn_=True,
        kBET_=False,
        pcr_=False,
        hvg_score_=False,
        cell_cycle_=False,
        n_cores=20,
        type_="embed",
    )

    results = results.rename_axis('metric', axis=0)
    results = results.rename(columns={results.columns[0]: 'score'})

    print(results)
    print(results.columns)
    print(results.index)

    results.to_csv(os.path.join(directory, "scib_metrics.csv"))

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    main(args.directory)