import os
import itertools
from collections import defaultdict

import pandas as pd
from cmmvae.constants import REGISTRY_KEYS as RK

def is_same_cell_type(row1, row2):
    return row1[RK.CELL_TYPE] == row2[RK.CELL_TYPE]

def compare_rows(row1, row2):
    compared = row1[RK.FILTER_CATEGORIES] == row2[RK.FILTER_CATEGORIES]
    differences = compared.index[~compared].tolist()
    n_differences = len(compared) - compared.sum()
    return n_differences, {RK.CONTEXT_A: row1[RK.CONTEXT_ID], RK.CONTEXT_B: row2[RK.CONTEXT_ID], RK.DIFFERENCES: differences}

def get_comparable_contexts(reference_dataframe: pd.DataFrame):
    
    comparable_contexts = defaultdict(list)  
    for idx1, idx2 in itertools.combinations(reference_dataframe.index, 2):

        if not is_same_cell_type(reference_dataframe.loc[idx1], reference_dataframe.loc[idx2]):
            continue

        n_differences, data = compare_rows(reference_dataframe.loc[idx1], reference_dataframe.loc[idx2])
        comparable_contexts[n_differences].append(data)
    print(comparable_contexts.keys())
    return comparable_contexts

def get_contexts(path: str):
    df = pd.read_csv(path)
    return get_comparable_contexts(df)

def save_contexts(path: str, contexts: dict[int, list[dict]]):
    for n_differences, data in contexts.items():
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(path, f"{n_differences}_differences.csv"), index=False)
        df.to_pickle(os.path.join(path, f"{n_differences}_differences.pkl"))

def main(path: str, file_name: str):
    context_data = get_contexts(os.path.join(path, file_name))
    save_contexts(path, context_data)

if __name__ == "__main__":
    main("/mnt/projects/debruinz_project/july2024_census_data/filtered_human", "context_references.csv")