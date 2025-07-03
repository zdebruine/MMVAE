import os
import glob

import pandas as pd
import scipy.sparse as sp

PATH = "/mnt/projects/debruinz_project/disease_study/datasets"
DISEASES = {"covid": "COVID-19", "crohn": "Crohn disease", "lung_adenocarcinoma": "lung adenocarcinoma"}

for disease, col_name in DISEASES.items():
    npz = sorted(glob.glob(os.path.join(PATH, f"withheld_filtered_{disease}*.npz")))
    pkl = sorted(glob.glob(os.path.join(PATH, f"withheld_filtered_{disease}*.pkl")))
    n = 0
    for mat, df in zip(npz, pkl):
        print("file:", df)
        data = sp.load_npz(mat)
        metadata = pd.read_pickle(df)
        print("data:", data.shape)
        print("metadata:", len(metadata))

        grouped = metadata.groupby(["cell_type", "tissue_general"], observed=True, sort=False)
        print("groups", len(grouped))

        for key, group in grouped:
            print("group:", key, len(group))
            disease_idx = group[group['disease'] == col_name].index.tolist()
            normal_idx = group[group['disease'] == 'normal'].index.tolist()
            print("disease:", len(disease_idx))
            print("healthy:", len(normal_idx))
            print("gidx:", group.index.tolist())

            if len(normal_idx) < 50 or len(disease_idx) < 50:
                continue

            group_data = data[group.index.tolist(), :]
            group_df = group.reset_index(drop=True)
            print("gidx after:", group_df.index.tolist())
            print("group data:", group_data.shape)
            print("group metadata:", len(group_df))
            sp.save_npz(os.path.join(PATH, f"withheld_{disease}_context_{n}.npz"), group_data)
            group_df.to_pickle(os.path.join(PATH, f"withheld_{disease}_context_{n}.pkl"))
            n += 1

