import scipy.sparse as sp
import pandas as pd
import torch

_CELL_CENSUS_COLUMN_NAMES = ["soma_joinid","dataset_id","assay","cell_type","development_stage","disease","donor_id","self_reported_ethnicity","sex","tissue","tissue_general"]

def convert_scipy_csr_torch(matrix):
    return torch.sparse_csr_tensor(matrix.indptr, matrix.indices, matrix.data, matrix.shape)

def load_data_and_metadata(data_file_path, metadata_file_path = None):
    matrix: sp.csr_matrix = sp.load_npz(data_file_path)
    if metadata:
        metadata = pd.read_csv(metadata_file_path, header=None, names=_CELL_CENSUS_COLUMN_NAMES)
    else:
        metadata = None
    return matrix, metadata

def split_data_and_metadata(data_file_path: str, metadata_file_path: pd.DataFrame, train_ratio: float):
    """
    Splits a csr_matrix and its corresponding metadata (pandas DataFrame) into training and validation sets based on a given ratio.
    Important:
        This function expects the metadata to have no header.
    :param data_file_path: The file path in which to load .npz.
    :param metadata_file_path: The file path to the pandas DataFrame containing metadata corresponding to the csr_matrix's rows.
    :param train_ratio: A float between 0 and 1 indicating the ratio of data to be used for training.
    :return: A tuple containing the training and validation sets for both the csr_matrix and the DataFrame.
    """
    
    matrix, metadata = load_data_and_metadata(data_file_path, metadata_file_path)

    # Calculate the split index
    split_index = int(matrix.shape[0] * train_ratio)

    # Split the matrix
    train_data = convert_scipy_csr_torch(matrix[:split_index])
    validation_data = convert_scipy_csr_torch(matrix[split_index:])

    # Split the metadata DataFrame
    train_metadata = metadata.iloc[:split_index]
    validation_metadata = metadata.iloc[split_index:]

    return (train_data, train_metadata), (validation_data, validation_metadata)