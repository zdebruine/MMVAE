import os

import pandas as pd
import scipy.sparse as sp

DISEASES = {"covid": "COVID-19", "crohn": "Crohn disease", "lung_adenocarcinoma": "lung adenocarcinoma"}

PATH = "/mnt/projects/debruinz_project/disease_study/"

full_files = [os.path.join(PATH, f"full/full_disease_study_metadata_{n}.pkl") for n in range(51,64)]

print("Loading Full Data")
full_df = pd.DataFrame()
for file in full_files:
    f = pd.read_pickle(file)
    full_df = pd.concat([full_df, f])

val_soma_ids = set(full_df['soma_joinid'])

for disease, col_name in DISEASES.items():
    print(f"Starting analysis for {disease}")
    selected_df = pd.read_pickle(os.path.join(PATH, f"selected/selected_{disease}_disease_study_metadata.pkl"))
    selected_counts = sp.load_npz(os.path.join(PATH, f"selected/selected_{disease}_disease_study_counts.npz"))

    selected_slice = selected_df[selected_df['soma_joinid'].isin(val_soma_ids)]
    selected_counts = selected_counts[selected_slice.index.tolist(), :]

    print(len(selected_slice))
    print(selected_counts.shape)

    sp.save_npz(os.path.join(PATH, f"selected/selected_{disease}_validation_counts.npz"), selected_counts) 
    selected_slice.to_pickle(os.path.join(PATH, f"selected/selected_{disease}_validation_metadata.pkl"))