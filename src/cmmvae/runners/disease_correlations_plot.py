"""
Get R^2 correlations between cis and cross species generations
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PATH = "/mnt/projects/debruinz_project/tony_boos/disease_study/correlations"
FILES = ["full_covid_", "full_crohn_", "full_lung_adenocarcinoma_", "filtered_covid_", "filtered_crohn_", "filtered_lung_adenocarcinoma_"]

for file in FILES:
    correlations = pd.read_csv(os.path.join(PATH, file + "correlations.csv"))

    width = max(8, min(2 * correlations["dataset_id"].nunique(), 19))  # Keep within a reasonable range
    height = max(6, min(2 * correlations["dataset_id"].nunique(), 10))
    plt.figure(figsize=(width, height))
    sns.violinplot(x="dataset_id", y="cis_over_true_and_cis", data=correlations, inner=None, color=".8")
    ax = sns.swarmplot(x="dataset_id", y="cis_over_true_and_cis", data=correlations, color="k", size=3)
    # ax = sns.stripplot(x="dataset_id", y="ratio", data=correlations, color="k", jitter=True, size=3)
    plt.title(f"Cross-Generation Ratio By Dataset For Cell and Tissue Type Groups")

    ticks = np.arange(0.94, 1.02, 0.02)

    plt.xlabel("Dataset-ID")
    # plt.xticks(rotation=90, ha="right", va="top")
    plt.xticks(rotation=15)

    plt.ylabel("Euclidean Distance Ratio")
    # plt.yticks(ticks)

    plt.tight_layout()
    plt.savefig(f"/mnt/projects/debruinz_project/tony_boos/disease_study/correlations/{file}_cis_over_true_cis_correlations.png")
    plt.clf()
    plt.close()
