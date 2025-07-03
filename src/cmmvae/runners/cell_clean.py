import os, glob

import numpy as np
import pandas as pd
import scipy.sparse as sp

PATH = "/mnt/projects/debruinz_project/july2024_census_data/"

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

human_npzs = glob.glob(os.path.join(PATH, "cells/human_*.npz"))
human_pkls = glob.glob(os.path.join(PATH, "cells/human_*.pkl"))
mouse_npzs = glob.glob(os.path.join(PATH, "cells/mouse_*.npz"))
mouse_pkls = glob.glob(os.path.join(PATH, "cells/mouse_*.pkl"))

human_npzs.sort()
human_pkls.sort()
mouse_npzs.sort()
mouse_pkls.sort()

for human_npz, human_pkl in zip(human_npzs, human_pkls):

    print(f"Processing {human_npz} and {human_pkl}")

    human_metadata = pd.read_pickle(human_pkl)
    human_data = sp.load_npz(human_npz)

    mask = np.ones(len(human_metadata), dtype=bool)
    for key, valid_values in conditions.items():
        mask &= human_metadata[key].isin(valid_values)

    idx = human_metadata[mask].index.tolist()
    sp.save_npz(human_npz, human_data[idx, :])

    human_metadata = human_metadata[mask]
    human_metadata = human_metadata.reset_index(drop=True)

    human_metadata.to_pickle(human_pkl)

for mouse_npz, mouse_pkl in zip(mouse_npzs, mouse_pkls):

    print(f"Processing {mouse_npz} and {mouse_pkl}")

    mouse_metadata = pd.read_pickle(mouse_pkl)
    mouse_data = sp.load_npz(mouse_npz)

    mask = np.ones(len(mouse_metadata), dtype=bool)
    for key, valid_values in conditions.items():
        mask &= mouse_metadata[key].isin(valid_values)

    idx = mouse_metadata[mask].index.tolist()
    sp.save_npz(mouse_npz, mouse_data[idx, :])

    mouse_metadata = mouse_metadata[mask]
    mouse_metadata = mouse_metadata.reset_index(drop=True)
    
    mouse_metadata.to_pickle(mouse_pkl)