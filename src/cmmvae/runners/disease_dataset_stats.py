import os
import glob
import pandas as pd

PATH = "/mnt/projects/debruinz_project/disease_study/datasets"

files = glob.glob(os.path.join(PATH, "full_*.pkl"))
files.sort()

stats_dfs = []

for f in files:
    df = pd.read_pickle(f)

    dataset_id = df["dataset_id"].iloc[0]

    grouped = df.groupby(["cell_type", "tissue_general", "disease"], observed=True).size().unstack(fill_value=0)
    disease_columns = [col for col in grouped.columns if col != "normal"]

    disease_col = disease_columns[0]
    grouped = grouped.rename(columns={disease_col: "disease"})

    if "normal" not in grouped.columns:
        grouped["normal"] = 0

    grouped["total_samples"] = grouped["normal"] + grouped["disease"]
    grouped = grouped.reset_index()
    grouped["dataset_id"] = dataset_id
    grouped["disease_name"] = disease_col

    stats_dfs.append(grouped)

stats = pd.concat(stats_dfs, ignore_index=True)
print(stats)

stats = stats[["disease_name", "dataset_id", "cell_type", "tissue_general", "normal", "disease", "total_samples"]]
stats.to_csv("/mnt/projects/debruinz_project/tony_boos/disease_study/dataset_stats.csv", index=False)