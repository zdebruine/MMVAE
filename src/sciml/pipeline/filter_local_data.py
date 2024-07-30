import os
import gc
import pandas as pd

base_dir = '/mnt/projects/debruinz_project/summer_census_data/full/'

def join_path(path):
    return os.path.join(base_dir, path)

def filter_local_data_main():
    
    print("filtering...", flush=True)
    expression_df = pd.read_pickle('/mnt/projects/debruinz_project/integration/tabula_sapiens/tabula_sapiens_metadata_expressions.pkl')
    
    print("generating paths...", flush=True)
    paths = [(join_path(f"human_counts_{i}.npz"), join_path(f"human_metadata_{i}.pkl")) for i in range(1, 61)]
    df_list = [pd.read_pickle(path) for _, path in paths]
    
    print("concatenating...", flush=True)
    # Concatenate all DataFrames in df_list
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("merging...", flush=True)
    # Merge with expression_df
    keys = [key for key in combined_df if key in expression_df]
    print(combined_df, expression_df, flush=True)
    print(keys, flush=True)
    merged_df = pd.merge(combined_df, expression_df[keys], on=keys, how='inner')
    
    if not merged_df.empty:
        print(merged_df)
    else:
        print("No data after merging.")
        
    merged_df.to_pickle("tabular_sapiens_metadata.pkl")

if __name__ == "__main__":
    filter_local_data_main()