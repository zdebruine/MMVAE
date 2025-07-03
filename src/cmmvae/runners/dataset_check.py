import os

import pandas as pd

DISEASES = {"covid": "COVID-19", "crohn": "Crohn disease", "lung_adenocarcinoma": "lung adenocarcinoma"}

PATH = "/mnt/projects/debruinz_project/disease_study/"

full_files = [os.path.join(PATH, f"full/full_disease_study_metadata_{n}.pkl") for n in range(51,64)]
filtered_files = [os.path.join(PATH, f"filtered/filtered_disease_study_metadata_{n}.pkl") for n in range(47,60)]

print("Loading Full Data")
full_df = pd.DataFrame()
for file in full_files:
    f = pd.read_pickle(file)
    full_df = pd.concat([full_df, f])

print("Loading Filtered Data")
filtered_df = pd.DataFrame()
for file in filtered_files:
    f = pd.read_pickle(file)
    filtered_df = pd.concat([filtered_df, f])

for disease, col_name in DISEASES.items():
    print(f"Starting analysis for {disease}")

    full_datasets = full_df[full_df['disease'] == col_name]['dataset_id'].unique().tolist()
    filtered_datasets = filtered_df[filtered_df['disease'] == col_name]['dataset_id'].unique().tolist()

    full_slice = full_df[full_df['dataset_id'].isin(full_datasets)]
    filtered_slice = filtered_df[filtered_df['dataset_id'].isin(filtered_datasets)]

    full = full_slice.groupby(["dataset_id", "disease"], observed=True).size().reset_index(name='num_samples')
    filtered = filtered_slice.groupby(["dataset_id", "disease"], observed=True).size().reset_index(name='num_samples')

    full.to_csv(f'/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_full_datasets_stats.csv', index=False)
    filtered.to_csv(f'/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_filtered_datasets_stats.csv', index=False)
