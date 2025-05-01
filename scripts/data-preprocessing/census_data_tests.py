import scipy.sparse as sp
import pandas as pd
import os
import glob

from data_processing_functions import extract_file_number

DIR = "/mnt/projects/debruinz_project/july2024_census_data/filtered"

human_data_files = glob.glob(os.path.join(DIR, "human*.npz"))
human_metadata_files = glob.glob(os.path.join(DIR, "human*.pkl"))
mouse_data_files = glob.glob(os.path.join(DIR, "mouse*.npz"))
mouse_metadata_files = glob.glob(os.path.join(DIR, "mouse*.pkl"))

human_data_files.sort(key=extract_file_number)
human_metadata_files.sort(key=extract_file_number)
mouse_data_files.sort(key=extract_file_number)
mouse_metadata_files.sort(key=extract_file_number)

assert len(human_data_files) == len(human_metadata_files) == len(mouse_data_files) == len(mouse_metadata_files)

for h_data_f, h_mdata_f, m_data_f, m_mdata_f in zip(human_data_files, human_metadata_files, mouse_data_files, mouse_metadata_files):
    human_data = sp.load_npz(h_data_f)
    human_metadata = pd.read_pickle(h_mdata_f)
    mouse_data = sp.load_npz(m_data_f)
    mouse_metadata = pd.read_pickle(m_mdata_f)

    try:
        assert human_data.shape[0] == human_metadata.shape[0]
        assert mouse_data.shape[0] == mouse_metadata.shape[0]
        assert human_data.shape[0] == mouse_data.shape[0]

        print(f"Sizes matched!")        

    except AssertionError as e:
        print(f"Assertion failed on size checks!!!\n{e}")

    try:
        hd_num = extract_file_number(h_data_f)
        hmd_num = extract_file_number(h_mdata_f)
        md_num = extract_file_number(m_data_f)
        mmd_num = extract_file_number(m_mdata_f)

        assert hd_num > 0 and hmd_num > 0
        assert md_num > 0 and mmd_num > 0
        assert hd_num == hmd_num == md_num == mmd_num
        print(f"File nums matched for filtered group number: {hd_num}")
    
    except AssertionError as e:
        print(f"Assertion failed on file number checks!!!\n{e}")