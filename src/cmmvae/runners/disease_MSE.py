import matplotlib.pyplot as plt
import pandas as pd
from cmmvae.runners.cross_generation import CrossGenerator
from torch.nn.functional import mse_loss

DISEASES = ["covid", "crohn", "lung_adenocarcinoma"]

full_model = CrossGenerator("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/full")
filtered_model = CrossGenerator("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/filtered")

for disease in DISEASES:

    data, metadata = full_model.get_data("/mnt/projects/debruinz_project/disease_study/selected", f"selected_{disease}_validation", sample=False)
    metadata = metadata.reset_index(drop=True)

    # assert len(metadata) > 100
    grouped = metadata.groupby(["cell_type", "tissue_general"], observed=True)

    mse = {
        "label": [],
        "full": [],
        "filtered": []
    }

    for key, group in grouped:

        indices = group.index.tolist()

        if len(indices) < 10:
            continue

        batch_data = data[indices, :]
        batch_metadata = metadata.loc[indices]

        full_out = full_model.get_cis_outputs(batch_data, batch_metadata, return_z=False)
        filtered_out = filtered_model.get_cis_outputs(batch_data, batch_metadata, return_z=False)

        mse["label"].append(" ".join(key))
        mse["full"].append(round(mse_loss(full_out, batch_data, reduction="mean").item(), 3))
        mse["filtered"].append(round(mse_loss(filtered_out, batch_data, reduction="mean").item(), 3))
    
    df = pd.DataFrame(mse)

    plt.figure(figsize=(14, 8))

    plt.scatter(x=df["full"], y=df["filtered"], s=2, alpha=0.5)

    plt.xlabel("Full Model")
    plt.xticks([0,0.1])

    plt.ylabel("Filtered Model")
    plt.yticks([0,0.1])

    plt.title(f"MSE of full and filtered models on {disease} by cell_type & tissue")

    plt.savefig(f"/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_full_vs_filtered_MSE.png", bbox_inches="tight")
    df.to_csv(f"/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_full_vs_filtered_MSE.csv", index=False)