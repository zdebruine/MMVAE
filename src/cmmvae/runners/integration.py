import os

import torch

import argparse as ap
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from cmmvae.runners.cross_generation import CrossGenerator
from cmmvae.constants import REGISTRY_KEYS as RK

PATH = "/mnt/projects/debruinz_project/july2024_census_data/subset/"

def plot_confusion_matrix(cm, classes, title, path):
    # Wrap your numpy CM in a DataFrame so the class names show up
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(12,10))                     # 1) make it bigger
    ax = sns.heatmap(
        df_cm,
        annot=True, fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'}               # 2) label the colorbar
    )
    ax.set_xlabel('Predicted cell type', fontsize=14)
    ax.set_ylabel('True cell type', fontsize=14)
    ax.set_title(title, fontsize=16)

    # 3) rotate x-labels + shrink font so they don’t overlap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()                             # 4) prevent cutting off labels
    plt.savefig(path, dpi=150)
    plt.close()

def integration_metrics(data, metadata):
    # Assume df is your DataFrame and 'category_column' contains the true labels
    le = LabelEncoder()
    true_labels = le.fit_transform(metadata['cell_type'])

    # Fit KMeans
    kmeans = KMeans(n_clusters=len(set(true_labels)), random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    # Compute the confusion matrix
    cm = skm.confusion_matrix(true_labels, cluster_labels)

    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create a mapping from cluster labels to true labels
    mapping = {cluster_label: true_label for cluster_label, true_label in zip(col_ind, row_ind)}

    # Map the cluster labels to the true labels
    mapped_cluster_labels = np.array([mapping[label] for label in cluster_labels])

    ari = skm.adjusted_rand_score(true_labels, mapped_cluster_labels)
    ami = skm.adjusted_mutual_info_score(true_labels, mapped_cluster_labels)
    homogeneity = skm.homogeneity_score(true_labels, mapped_cluster_labels)
    completeness = skm.completeness_score(true_labels, mapped_cluster_labels)
    v_measure = skm.v_measure_score(true_labels, mapped_cluster_labels)
    cm = skm.confusion_matrix(true_labels, mapped_cluster_labels)

    return ari, ami, homogeneity, completeness, v_measure, cm, le.classes_

def main(directory: str):
    model = CrossGenerator(directory)

    output_dir = os.path.join(directory, "integration")
    os.makedirs(output_dir, exist_ok=True)

    human_data = model._convert_to_tensor(sp.load_npz(os.path.join(PATH, "human_counts_15.npz")))
    human_metadata = pd.read_pickle(os.path.join(PATH, "human_metadata_15.pkl"))
    human_metadata["species"] = RK.HUMAN

    mouse_data = model._convert_to_tensor(sp.load_npz(os.path.join(PATH, "mouse_counts_15.npz")))
    mouse_metadata = pd.read_pickle(os.path.join(PATH, "mouse_metadata_15.pkl"))
    mouse_metadata["species"] = RK.MOUSE

    human_z = model._get_z(human_data, RK.HUMAN)
    mouse_z = model._get_z(mouse_data, RK.MOUSE)

    integration = pd.DataFrame(columns=["ari", "ami", "homogeneity", "completeness", "v_measure"], index=[RK.HUMAN, RK.MOUSE, "multispecies"])

    ari, ami, homogeneity, completeness, v_measure, cm, labels = integration_metrics(human_z.cpu().numpy(), human_metadata)

    plot_confusion_matrix(cm, labels, 'Confusion Matrix (Human Only)', os.path.join(output_dir, "human_cm.png"))

    integration.loc[RK.HUMAN] = [ari, ami, homogeneity, completeness, v_measure]

    ari, ami, homogeneity, completeness, v_measure, cm, labels = integration_metrics(mouse_z.cpu().numpy(), mouse_metadata)

    plot_confusion_matrix(cm, labels, 'Confusion Matrix (Mouse Only)', os.path.join(output_dir, "mouse_cm.png"))

    integration.loc[RK.MOUSE] = [ari, ami, homogeneity, completeness, v_measure]

    latents = torch.cat([human_z, mouse_z], dim=0).cpu().numpy()
    latent_md = pd.concat([human_metadata, mouse_metadata], axis=0, ignore_index=True)

    idx = np.random.permutation(latents.shape[0])

    latents = latents[idx]
    latent_md = latent_md.iloc[idx]
    latent_md = latent_md.reset_index(drop=True)

    ari, ami, homogeneity, completeness, v_measure, cm, labels = integration_metrics(latents, latent_md)

    plot_confusion_matrix(cm, labels, 'Confusion Matrix (Multispecies)', os.path.join(output_dir, "multispecies_cm.png"))

    integration.loc[RK.MOUSE] = [ari, ami, homogeneity, completeness, v_measure]

    integration = integration.reset_index().rename(columns={'index': 'species'})

    integration.to_csv(os.path.join(output_dir, "integration_scores.csv"), index=False)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    main(args.directory)