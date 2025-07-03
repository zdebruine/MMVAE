import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DISEASES = ["covid", "crohn", "lung_adenocarcinoma"]

ticks = np.arange(0, 0.105, 0.01)

for disease in DISEASES:
    df = pd.read_csv(f"/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_full_vs_filtered_MSE.csv")

    plt.figure(figsize=(14, 8))

    plt.scatter(x=df["full"], y=df["filtered"], s=10, alpha=0.5)

    plt.plot(ticks, ticks, color='red')

    plt.xlabel("Full Model")
    plt.xticks(ticks)

    plt.ylabel("Filtered Model")
    plt.yticks(ticks)

    plt.title(f"MSE of full and filtered models on {disease} by cell_type & tissue")

    plt.savefig(f"/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_full_vs_filtered_MSE.png", bbox_inches="tight")