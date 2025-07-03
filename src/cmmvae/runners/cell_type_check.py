import os

import pandas as pd

DISEASES = {"covid": "COVID-19", "crohn": "Crohn disease", "lung_adenocarcinoma": "lung adenocarcinoma"}

PATH = "/mnt/projects/debruinz_project/disease_study/"

full_files = [os.path.join(PATH, f"full/full_disease_study_metadata_{n}.pkl") for n in range(1,64)]
filtered_files = [os.path.join(PATH, f"filtered/filtered_disease_study_metadata_{n}.pkl") for n in range(1,60)]
# selected_files = [os.path.join(PATH, f"selected/selected_{disease}_disease_study_metadata.pkl") for disease in DISEASES]

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
    selected_df = pd.read_pickle(os.path.join(PATH, f"selected/selected_{disease}_disease_study_metadata.pkl"))
    full_cells = pd.DataFrame(
        {"cell_type":
            full_df[full_df['disease'] == col_name]['cell_type'].unique().tolist()
        }
    )
    filtered_cells = pd.DataFrame(
        {"cell_type":
            filtered_df[filtered_df['disease'] == col_name]['cell_type'].unique().tolist()
        }
    )
    selected_cells = pd.DataFrame(
        {"cell_type":
            selected_df['cell_type'].unique().tolist()
        }
    )
    full_cells.to_csv(f'/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_full_cells.csv', index=False)
    filtered_cells.to_csv(f'/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_filtered_cells.csv', index=False)
    selected_cells.to_csv(f'/mnt/projects/debruinz_project/tony_boos/disease_study/{disease}_selected_cells.csv', index=False)
