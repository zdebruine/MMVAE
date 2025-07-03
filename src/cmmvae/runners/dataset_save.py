import os

import pandas as pd
import scipy.sparse as sp

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

full_valid_datasets = {disease: [] for disease in DISEASES.keys()}
filtered_valid_datasets = {disease: [] for disease in DISEASES.keys()}

for disease, col_name in DISEASES.items():
    print(f"Starting analysis for {disease}")

    full_datasets = full_df[full_df['disease'] == col_name]['dataset_id'].unique().tolist()
    filtered_datasets = filtered_df[filtered_df['disease'] == col_name]['dataset_id'].unique().tolist()

    full_slice = full_df[full_df['dataset_id'].isin(full_datasets)]
    filtered_slice = filtered_df[filtered_df['dataset_id'].isin(filtered_datasets)]

    full = full_slice.groupby(["dataset_id", "disease"], observed=True).size().reset_index(name='num_samples')
    filtered = filtered_slice.groupby(["dataset_id", "disease"], observed=True).size().reset_index(name='num_samples')

    for dataset in full_datasets:
        datasets = full[full["dataset_id"] == dataset]
        dataset_diseases = datasets['disease'].values
        if "normal" in dataset_diseases and col_name in dataset_diseases:
            if all(datasets['num_samples'].apply(lambda x: x >= 100)):
                full_valid_datasets[disease].append(dataset)

    for dataset in filtered_datasets:
        datasets = filtered[filtered["dataset_id"] == dataset]
        dataset_diseases = datasets['disease'].values
        if "normal" in dataset_diseases and col_name in dataset_diseases:
            if all(datasets['num_samples'].apply(lambda x: x >= 100)):
                filtered_valid_datasets[disease].append(dataset)

full_files = {
    disease: {
        dataset: {
            "mat": sp.csr_matrix((0, 60530)),
            "df": pd.DataFrame(),
        }for dataset in datasets
    }for disease, datasets in full_valid_datasets.items() 
}

for n in range(51,64):
    df = pd.read_pickle(os.path.join(PATH, f"full/full_disease_study_metadata_{n}.pkl"))
    npz = sp.load_npz(os.path.join(PATH, f"full/full_disease_study_counts_{n}.npz"))
    for disease, datasets in full_valid_datasets.items():
        for dataset in datasets:
            idx = df[df['dataset_id'] == dataset].index.tolist()
            full_files[disease][dataset]["mat"] = sp.vstack([full_files[disease][dataset]["mat"], npz[idx, :]])
            full_files[disease][dataset]["df"] = pd.concat([full_files[disease][dataset]["df"], df.loc[idx]]).reset_index(drop=True)

for disease, datasets in full_files.items():
    for dataset_id, data in datasets.items():
        sp.save_npz(os.path.join(PATH, f"datasets/full_{disease}_{dataset_id}.npz"), data["mat"])
        data["df"].to_pickle(os.path.join(PATH, f"datasets/full_{disease}_{dataset_id}.pkl"))
    
filtered_files = {
    disease: {
        dataset: {
            "mat": sp.csr_matrix((0, 60530)),
            "df": pd.DataFrame(),
        }for dataset in datasets
    }for disease, datasets in filtered_valid_datasets.items() 
}

for n in range(47,60):
    df = pd.read_pickle(os.path.join(PATH, f"filtered/filtered_disease_study_metadata_{n}.pkl"))
    npz = sp.load_npz(os.path.join(PATH, f"filtered/filtered_disease_study_counts_{n}.npz"))
    for disease, datasets in filtered_valid_datasets.items():
        for dataset in datasets:
            idx = df[df['dataset_id'] == dataset].index.tolist()
            filtered_files[disease][dataset]["mat"] = sp.vstack([filtered_files[disease][dataset]["mat"], npz[idx, :]])
            filtered_files[disease][dataset]["df"] = pd.concat([filtered_files[disease][dataset]["df"], df.loc[idx]]).reset_index(drop=True)

for disease, datasets in filtered_files.items():
    for dataset_id, data in datasets.items():
        sp.save_npz(os.path.join(PATH, f"datasets/filtered_{disease}_{dataset_id}.npz"), data["mat"])
        data["df"].to_pickle(os.path.join(PATH, f"datasets/filtered_{disease}_{dataset_id}.pkl"))