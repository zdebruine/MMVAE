import os

import pandas as pd

from cmmvae.constants import REGISTRY_KEYS as RK

# import cellxgene_census as cxg

# CENSUS_VERSION = "2024-07-01"

# census = cxg.open_soma(census_version=CENSUS_VERSION)

diseases = ["COVID-19", "lung adenocarcinoma", "Crohn disease"]

# for disease in diseases:
#     obs = cxg.get_obs(
#         census=census,
#         organism="Homo sapiens",
#         value_filter=f"disease == '{disease}' and is_primary_data == True"
#     )
#     # df = obs.read().concat().to_pandas()
#     obs.to_csv(f"/mnt/projects/debruinz_project/tony_boos/full_data_{disease.replace(' ', '_')}_summary.csv", index=False)

# census.close()

# PATH = "/mnt/projects/debruinz_project/july2024_census_data/full/"

# files = [os.path.join(PATH, f"human_metadata_{n}.pkl") for n in range(1,90)]

# df = pd.DataFrame()
# for file in files:
#     f = pd.read_pickle(file)
#     df = pd.concat([df, f])


# group_by_columns = ["tissue_general", "cell_type", "donor_id", "dataset_id"]

def group_and_save(df):
    summary = df.groupby(["cell_type"]).agg(
        num_samples=("num_samples", "sum"),
        num_datasets=("dataset_id", lambda x: x.nunique()),
        num_tissues=("tissue_general", lambda x: x.nunique()),
        num_donors=("donor_id", lambda x: x.nunique()),
        num_dev_stages=("dev_stage", lambda x: x.nunique()),
        num_sexes=("sex", lambda x: x.nunique()),
    ).reset_index()

    summary = summary[["num_samples", "cell_type", "num_datasets", "num_tissues", "num_donors", "num_dev_stages", "num_sexes"]]
    # unique_combos = unique_combos[cols]
    summary.to_csv(f"/mnt/projects/debruinz_project/tony_boos/full_data_{disease.replace(' ', '_')}_cell_stats.csv", index=False)


for disease in diseases:
    # df_disease = df[df["disease"] == disease]
    # Get unique combinations and count occurrences in the full dataset
    df = pd.read_csv(f"/mnt/projects/debruinz_project/tony_boos/full_data_{disease.replace(' ', '_')}_summary.csv")
    group_and_save(df)
    # unique_combos = df.groupby(RK.FILTER_CATEGORIES, as_index=False).size().rename(columns={"size": "num_samples"})
    # # unique_combos["disease"] = disease 
    # cols = ["num_samples"] + RK.FILTER_CATEGORIES
    # unique_combos = unique_combos[cols]
    # unique_combos.to_csv(f"/mnt/projects/debruinz_project/tony_boos/full_data_{disease.replace(' ', '_')}_summary.csv", index=False)

# output = pd.DataFrame(columns=["context_id", "num_samples"] + RK.FILTER_CATEGORIES)
# grouped = df.groupby(RK.FILTER_CATEGORIES)
# for cid, (id, idx) in enumerate(grouped.groups.items(), start=1):
#     row = {"context_id": [cid], "num_samples": [len(idx)]}
#     row.update({cat: val for cat, val in zip(RK.FILTER_CATEGORIES, id)})
#     # for cat, val in zip(RK.FILTER_CATEGORIES, id):
#     #     row[cat] = [val]
#     output = pd.concat([output, pd.DataFrame(row)])

# 1. Overall disease summary
# disease_summary = df.groupby("disease").agg(
#     total_samples=("disease", "size"),          # total samples per disease
#     total_datasets=("dataset_id", "nunique"),     # unique datasets per disease
#     total_donors=("donor_id", "nunique")          # unique donors per disease
# ).reset_index()

# # 2. Unique tissue and cell type counts per disease
# unique_tissue_cell = df.groupby("disease").agg(
#     unique_tissues=("tissue_general", "nunique"),
#     unique_cell_types=("cell_type", "nunique")
# ).reset_index()

# # Merge overall disease summary with unique tissue/cell type counts
# final_df = pd.merge(disease_summary, unique_tissue_cell, on="disease")
# final_df = final_df[["total_samples", "total_datasets", "total_donors", "unique_cell_types", "unique_tissues", "disease"]]
# final_df = final_df.sort_values(by="disease")


# final_df.to_csv("/mnt/projects/debruinz_project/tony_boos/full_data_disease_summary.csv", index=False)