import os

import pandas as pd

HUMAN_DATASET = "d6505c89-c43d-4c28-8c4f-7351a5fd5528"
MOUSE_DATASET = "23f77ae6-10af-4307-b136-76b26654ae7d"

PATH = "/mnt/projects/debruinz_project/july2024_census_data/"

human_train_files = [os.path.join(PATH, f"subset/human_metadata_{n}.pkl") for n in range(1,14)]
mouse_train_files = [os.path.join(PATH, f"subset/mouse_metadata_{n}.pkl") for n in range(1,14)]

human_files = [os.path.join(PATH, f"full/human_metadata_{n}.pkl") for n in range(1,90)]
mouse_files = [os.path.join(PATH, f"full/mouse_metadata_{n}.pkl") for n in range(1,34)]

human_train_ids = set()
mouse_train_ids = set()

train_dfs = []
print("Getting Human Train IDs")
for file in human_files:
    f = pd.read_pickle(file)
    train_dfs.append(f)
df = pd.concat(train_dfs, ignore_index=True)
human_train_ids.update(set(df['soma_joinid']))

train_dfs = []
print("Getting Mouse Train IDs")
for file in mouse_files:
    f = pd.read_pickle(file)
    train_dfs.append(f)
df = pd.concat(train_dfs, ignore_index=True)
mouse_train_ids.update(set(df['soma_joinid']))

print("Loading Human Data")
human_dfs = []
for file in human_files:
    f = pd.read_pickle(file)
    human_df = pd.concat([human_df, f])

jin_human_df = human_df[human_df['dataset_id'] == HUMAN_DATASET]
human_counts = jin_human_df.groupby('cell_type', observed=True).size().reset_index(name='count')
human_counts.to_csv('/mnt/projects/debruinz_project/tony_boos/multispecies_study/jin_human_cell_types_subset.csv', index=False)

print("Loading Mouse Data")
mouse_df = pd.DataFrame()
for file in mouse_files:
    f = pd.read_pickle(file)
    mouse_df = pd.concat([mouse_df, f])

jin_mouse_df = mouse_df[mouse_df['dataset_id'] == MOUSE_DATASET]
mouse_counts = jin_mouse_df.groupby('cell_type', observed=True).size().reset_index(name='count')
mouse_counts.to_csv('/mnt/projects/debruinz_project/tony_boos/multispecies_study/jin_mouse_cell_types_subset.csv', index=False)


