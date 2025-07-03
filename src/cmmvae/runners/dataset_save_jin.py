import os

import pandas as pd
import scipy.sparse as sp

HUMAN_DATASET = "d6505c89-c43d-4c28-8c4f-7351a5fd5528"
MOUSE_DATASET = "23f77ae6-10af-4307-b136-76b26654ae7d"

PATH = "/mnt/projects/debruinz_project/july2024_census_data/"

human_train_files = [os.path.join(PATH, f"subset/human_metadata_{n}.pkl") for n in range(1,14)]
mouse_train_files = [os.path.join(PATH, f"subset/mouse_metadata_{n}.pkl") for n in range(1,14)]

human_files = [(os.path.join(PATH, f"full/human_counts_{n}.npz"), os.path.join(PATH, f"full/human_metadata_{n}.pkl")) for n in range(1,90)]
mouse_files = [(os.path.join(PATH, f"full/mouse_counts_{n}.npz"), os.path.join(PATH, f"full/mouse_metadata_{n}.pkl")) for n in range(1,34)]

train_dfs = []
print("Getting Human Train IDs")
for file in human_train_files:
    f = pd.read_pickle(file)
    train_dfs.append(f)
df = pd.concat(train_dfs, ignore_index=True)

df = df[df['dataset_id'] == HUMAN_DATASET]
human_train_ids = set(df['soma_joinid'])

human_counts = df.groupby('cell_type', observed=True).size().reset_index(name='count')
human_counts.to_csv('/mnt/projects/debruinz_project/tony_boos/multispecies_study/jin_human_cell_types_subset_train.csv', index=False)

train_dfs = []
print("Getting Mouse Train IDs")
for file in mouse_train_files:
    f = pd.read_pickle(file)
    train_dfs.append(f)
df = pd.concat(train_dfs, ignore_index=True)

df = df[df['dataset_id'] == MOUSE_DATASET]
mouse_train_ids = set(df['soma_joinid'])

mouse_counts = df.groupby('cell_type', observed=True).size().reset_index(name='count')
mouse_counts.to_csv('/mnt/projects/debruinz_project/tony_boos/multispecies_study/jin_mouse_cell_types_subset_train.csv', index=False)

print("Loading Human Data")
human_counts = []
human_dfs = []
for npz, pkl in human_files:
    print(npz, pkl)
    full_df = pd.read_pickle(pkl)
    jin_df = full_df[(full_df['dataset_id'] == HUMAN_DATASET) & ~(full_df["soma_joinid"].isin(human_train_ids))]

    if jin_df.empty:
        continue
    
    idx = jin_df.index.tolist()
    jin_df = jin_df.reset_index(drop=True)

    counts = sp.load_npz(npz)
    counts = counts[idx, :]

    human_counts.append(counts)
    human_dfs.append(jin_df)

human_mat = sp.vstack(human_counts, format='csr')
human_df = pd.concat(human_dfs, ignore_index=True)

if not human_mat.shape[0] == len(human_df):
    print("Size error in human!")

sp.save_npz(os.path.join(PATH, "jin/human_counts_subset.npz"), human_mat)
human_df.to_pickle(os.path.join(PATH, "jin/human_metadata_subset.pkl"))
human_counts = human_df.groupby('cell_type', observed=True).size().reset_index(name='count')
human_counts.to_csv('/mnt/projects/debruinz_project/tony_boos/multispecies_study/jin_human_cell_types_subset.csv', index=False)

print("Loading Mouse Data")
mouse_counts = []
mouse_dfs = []
for npz, pkl in mouse_files:
    print(npz, pkl)
    full_df = pd.read_pickle(pkl)
    jin_df = full_df[(full_df['dataset_id'] == MOUSE_DATASET) & ~(full_df["soma_joinid"].isin(mouse_train_ids))]

    if jin_df.empty:
        continue
    
    idx = jin_df.index.tolist()
    jin_df = jin_df.reset_index(drop=True)

    counts = sp.load_npz(npz)
    counts = counts[idx, :]

    mouse_counts.append(counts)
    mouse_dfs.append(jin_df)

mouse_mat = sp.vstack(mouse_counts, format='csr')
mouse_df = pd.concat(mouse_dfs, ignore_index=True)

if not mouse_mat.shape[0] == len(mouse_df):
    print("Size error in mouse!")

sp.save_npz(os.path.join(PATH, "jin/mouse_counts_subset.npz"), mouse_mat)
mouse_df.to_pickle(os.path.join(PATH, "jin/mouse_metadata_subset.pkl"))
mouse_counts = mouse_df.groupby('cell_type', observed=True).size().reset_index(name='count')
mouse_counts.to_csv('/mnt/projects/debruinz_project/tony_boos/multispecies_study/jin_mouse_cell_types_subset.csv', index=False)