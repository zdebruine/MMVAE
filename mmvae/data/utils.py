import scipy.sparse as sp
import pandas as pd
import torch

_CELL_CENSUS_COLUMN_NAMES = ["soma_joinid","dataset_id","assay","cell_type","development_stage","disease","donor_id","self_reported_ethnicity","sex","tissue","tissue_general"]

def split_data_and_metadata(data_file_path: str, metadata_file_path: str = None, train_ratio: float = None, header=False):
    """
    Splits a csr_matrix and its corresponding metadata (pandas DataFrame) into training and validation sets based on a given ratio.
    Important:
        This function expects the metadata to have no header.
    :param data_file_path: The file path in which to load .npz.
    :param metadata_file_path: The file path to the pandas DataFrame containing metadata corresponding to the csr_matrix's rows.
    :param train_ratio: A float between 0 and 1 indicating the ratio of data to be used for training. If None or not 0 < float < 1 then no splitting
    :return: A tuple containing the training and validation sets for both the csr_matrix and the DataFrame.
    """
    matrix: sp.csr_matrix = sp.load_npz(data_file_path)
    split_data = train_ratio is not None and 0 < train_ratio < 1
    
    if metadata_file_path is not None:
        if not header:
            metadata = pd.read_csv(metadata_file_path, header=None, names=_CELL_CENSUS_COLUMN_NAMES)
        else:
            metadata = pd.read_csv(metadata_file_path)
    else:
        metadata = None
        
    if metadata is not None and matrix.shape[0] != len(metadata):
        raise ValueError(f"The number of rows in the matrix and metadata must match: ({matrix.shape[0]} | {len(metadata)})")
    
    if split_data:
        # Calculate the split index
        split_index = int(matrix.shape[0] * train_ratio)
        test_data = matrix[split_index:]
        train_data = matrix[:split_index]
        if metadata:
            train_metadata = metadata.iloc[:split_index]
            test_metadata = metadata.iloc[split_index:]
        else:
            train_metadata = None
            test_metadata = None
    else:
        train_data = matrix
        train_metadata = metadata
        test_data = None
        test_metadata = None
    
    train_data = torch.sparse_csr_tensor(train_data.indptr, train_data.indices, train_data.data, train_data.shape)
    if test_data is not None:
        test_data = torch.sparse_csr_tensor(test_data.indptr, test_data.indices, test_data.data, test_data.shape)
    
    if test_data == None and test_metadata == None:
        return ((train_data, train_metadata),)
    else:
        return ((train_data, train_metadata), (test_data, test_metadata))   