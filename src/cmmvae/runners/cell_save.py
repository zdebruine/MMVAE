import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

def clean_cell_key(key: str):
    return key.replace(" ", "_").replace("/", "_").replace(".", "_")

PATH = "/mnt/projects/debruinz_project/july2024_census_data/"

human_train_files = [os.path.join(PATH, f"subset/human_metadata_{n}.pkl") for n in range(1,14)]
mouse_train_files = [os.path.join(PATH, f"subset/mouse_metadata_{n}.pkl") for n in range(1,14)]

human_files = [(os.path.join(PATH, f"full/human_counts_{n}.npz"), os.path.join(PATH, f"full/human_metadata_{n}.pkl")) for n in range(1,90)]
mouse_files = [(os.path.join(PATH, f"full/mouse_counts_{n}.npz"), os.path.join(PATH, f"full/mouse_metadata_{n}.pkl")) for n in range(1,34)]

condition_paths = {
    "assay": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_assay.csv",
    "dataset_id": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_dataset_id.csv",
    "dev_stage": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_dev_stage.csv",
    "disease": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_disease.csv",
    "donor_id": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_donor_id.csv",
    "sex": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_sex.csv",
    "suspension_type": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_suspension_type.csv",
    "tissue_general": "/mnt/projects/debruinz_project/tony_boos/3m_expressions/unique_expression_tissue_general.csv",
}

conditions = {
    key: set(pd.read_csv(path, header=None, names=[key])[key])
    for key, path in condition_paths.items()
}

train_dfs = []
print("Getting Human Train IDs")
for file in human_train_files:
    f = pd.read_pickle(file)
    train_dfs.append(f)
df = pd.concat(train_dfs, ignore_index=True)

human_train_ids = set(df['soma_joinid'])

human_train_counts = df.groupby('cell_type', observed=True).size().reset_index(name='count')

train_dfs = []
print("Getting Mouse Train IDs")
for file in mouse_train_files:
    f = pd.read_pickle(file)
    train_dfs.append(f)
df = pd.concat(train_dfs, ignore_index=True)

mouse_train_ids = set(df['soma_joinid'])

mouse_train_counts = df.groupby('cell_type', observed=True).size().reset_index(name='count')

print("Loading Human Data")
human_dfs = []
for i, (_, pkl) in enumerate(human_files, start=1):

    df = pd.read_pickle(pkl)
    df = df[~(df["soma_joinid"].isin(human_train_ids))]

    mask = np.ones(len(df), dtype=bool)
    for key, valid_values in conditions.items():
        mask &= df[key].isin(valid_values)
    
    df = df[mask]

    df["chunk_source"] = i
    df = df.reset_index().rename(columns={'index': 'chunk_index'})
    human_dfs.append(df)

human_df = pd.concat(human_dfs, ignore_index=True)

print("Loading Mouse Data")
mouse_dfs = []
for i, (_, pkl) in enumerate(mouse_files, start=1):

    df = pd.read_pickle(pkl)
    df = df[~(df["soma_joinid"].isin(mouse_train_ids))]

    mask = np.ones(len(df), dtype=bool)
    for key, valid_values in conditions.items():
        mask &= df[key].isin(valid_values)

    df = df[mask]

    df["chunk_source"] = i
    df = df.reset_index().rename(columns={'index': 'chunk_index'})
    mouse_dfs.append(df)

mouse_df = pd.concat(mouse_dfs, ignore_index=True)

human_counts = human_df['cell_type'].value_counts()
mouse_counts = mouse_df['cell_type'].value_counts()

set_found = False
head = 15
while not set_found:
    top_human = human_counts.head(head)
    top_mouse = mouse_counts.head(head)

    common = set(top_human.index) & set(top_mouse.index)

    if len(common) >= 15:
        set_found = True
        human_df = human_df[human_df['cell_type'].isin(common)].reset_index(drop=True)
        mouse_df = mouse_df[mouse_df['cell_type'].isin(common)].reset_index(drop=True)
        break
    else:
        head += 1

human_counts = human_df.groupby('cell_type', observed=True).size().reset_index(name='count')
human_counts.to_csv(os.path.join(PATH, f"cells/human_counts.csv"), index=False)

mouse_counts = mouse_df.groupby('cell_type', observed=True).size().reset_index(name='count')
mouse_counts.to_csv(os.path.join(PATH, f"cells/mouse_counts.csv"), index=False)

human_train_counts = human_train_counts[human_train_counts['cell_type'].isin(human_counts['cell_type'])].reset_index(drop=True)
mouse_train_counts = mouse_train_counts[mouse_train_counts['cell_type'].isin(mouse_counts['cell_type'])].reset_index(drop=True)

human_train_counts.to_csv(os.path.join(PATH, f"cells/human_train_counts.csv"), index=False)
mouse_train_counts.to_csv(os.path.join(PATH, f"cells/mouse_train_counts.csv"), index=False)

print("Loading Human Data")
human_counts = []
human_dfs = []
for i, (npz, pkl) in enumerate(human_files, start=1):

    idx = human_df[human_df['chunk_source'] == i]['chunk_index'].tolist()
    counts = sp.load_npz(npz)
    counts = counts[idx, :]

    df = pd.read_pickle(pkl)
    df = df.iloc[idx].reset_index(drop=True)

    human_counts.append(counts)
    human_dfs.append(df)

human_mat = sp.vstack(human_counts, format='csr')
human_df = pd.concat(human_dfs, ignore_index=True)

human_counts.clear()
human_dfs.clear()

if not human_mat.shape[0] == len(human_df):
    print("Size error in human!")

print("Loading Mouse Data")
mouse_counts = []
mouse_dfs = []
for i, (npz, pkl) in enumerate(mouse_files, start=1):

    idx = mouse_df[mouse_df['chunk_source'] == i]['chunk_index'].tolist()
    counts = sp.load_npz(npz)
    counts = counts[idx, :]

    df = pd.read_pickle(pkl)
    df = df.iloc[idx].reset_index(drop=True)

    mouse_counts.append(counts)
    mouse_dfs.append(df)

mouse_mat = sp.vstack(mouse_counts, format='csr')
mouse_df = pd.concat(mouse_dfs, ignore_index=True)

mouse_counts.clear()
mouse_dfs.clear()

if not mouse_mat.shape[0] == len(mouse_df):
    print("Size error in mouse!")

for cell_type, group in human_df.groupby('cell_type', observed=True):

    key = clean_cell_key(cell_type)

    idx = group.index.tolist()
    mat = human_mat[idx, :]

    df = group.reset_index(drop=True)

    sp.save_npz(os.path.join(PATH, f"cells/human_{key}.npz"), mat)
    df.to_pickle(os.path.join(PATH, f"cells/human_{key}.pkl"))

for cell_type, group in mouse_df.groupby('cell_type', observed=True):

    key = clean_cell_key(cell_type)

    idx = group.index.tolist()
    mat = mouse_mat[idx, :]

    df = group.reset_index(drop=True)

    sp.save_npz(os.path.join(PATH, f"cells/mouse_{key}.npz"), mat)
    df.to_pickle(os.path.join(PATH, f"cells/mouse_{key}.pkl"))
