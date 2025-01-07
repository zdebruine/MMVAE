import os
import re
import csv
import numpy as np
import pandas as pd
import pandas.core.groupby.generic as gb
from collections import defaultdict
from data_processing_functions import extract_file_number
from cmmvae.constants import REGISTRY_KEYS as RK

TRAIN_DATA_FILES = set(range(1, 14))
MIN_SAMPLE_SIZE = 100
MAX_SAMPLE_SIZE = 1000
SPECIES_SAMPLE_SIZE = {"human": 60530, "mouse": 52437}

def get_train_data_ids(files: tuple[str]) -> set[int]:
    
    combined_df = pd.DataFrame()
    
    for file in files:
        df = pd.read_pickle(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return set(combined_df["soma_joinid"])

def filter_train_ids(df: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    filtered_df = df[~(df["soma_joinid"].isin(ids))]
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def filter_and_sort_train_files(unfiltered_files: tuple[str]) -> tuple[str]:

    filtered_files = [
        file for file in unfiltered_files 
        if (match := re.search(r'(\d+)\.pkl$', file)) and int(match.group(1)) in TRAIN_DATA_FILES
    ]
    filtered_files.sort(key=extract_file_number)
    return tuple(filtered_files)

def load_and_merge_metadata(files: tuple[str]) -> pd.DataFrame:
    
    merged_df = pd.DataFrame()
    for file in files:
        df = pd.read_pickle(file)
        df["chunk_source"] = extract_file_number(file)
        df = df.reset_index(drop=False)
        df.rename(columns={"index": "data_index"}, inplace=True)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    return merged_df

def filter_into_groups(dfs: dict[str, pd.DataFrame]):
    
    grouped = {}
    for specie, data in dfs.items():
        grouped[specie] = data.groupby(RK.FILTER_CATEGORIES)

    return grouped

def validate_and_sample_groups(data_groups: dict[str, gb.DataFrameGroupBy], primary_species: str = None):
    
    valid_groups = defaultdict(dict)
    if primary_species is not None:
        main_df = data_groups.pop(primary_species)
    else:
        primary_species, main_df = data_groups.popitem()

    for gid, idxes in main_df.groups.items():
        if len(idxes) < MIN_SAMPLE_SIZE:
            continue
        elif all(
            gid in group.groups.keys() and len(group.groups[gid]) >= MIN_SAMPLE_SIZE
            for group in data_groups.values()
        ):
            sample_size = min(
                [len(idxes), MAX_SAMPLE_SIZE] + [len(group.groups[gid]) for group in data_groups.values()]
            )

            valid_groups[gid][primary_species] = np.random.choice(idxes, sample_size, replace= False)
            for specie, group in data_groups.items():
                valid_groups[gid][specie] = np.random.choice(group.groups[gid], sample_size, replace= False)
            
    return valid_groups

def save_grouped_data(groups: dict[tuple[str], dict[str, np.ndarray]], dfs: dict[str, pd.DataFrame], save_dir: str):
    
    with open(os.path.join(save_dir, "group_references.csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerow(["group_id", "num_samples"] + RK.FILTER_CATEGORIES)
        for i, gid in enumerate(groups.keys(), start=1):
            for specie, idx in groups[gid].items():                
                df = dfs[specie].iloc[idx]
                df["group_id"] = i
                df["num_samples"] = len(idx)
                df = df.sort_values("chunk_source")
                df.to_pickle(os.path.join(save_dir, f"{specie}_filtered_{i}.pkl"))
            writer.writerow([i, len(idx)] + list(gid))
    file = pd.read_csv(os.path.join(save_dir, "group_references.csv"))
    file.to_pickle(os.path.join(save_dir, "group_references.pkl"))
