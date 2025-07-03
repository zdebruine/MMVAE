import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

def save_as_h5(
        matrix: sp.csr_matrix,
        metadata: pd.DataFrame,
        gene_metadata: pd.DataFrame,
        save_dir: str,
        filename: str
) -> None:
    """
    Save a sparse matrix and its metadata to an HDF5 file.
    Args:
        matrix (sp.csr_matrix): Sparse matrix to save.
        metadata (pd.DataFrame): Metadata associated with the matrix.
        gene_metadata (pd.DataFrame): Metadata associated with the genes.
        save_dir (str): Directory to save the HDF5 file.
        filename (str): Name of the HDF5/CSV file (without extension).
    """
    
    if type(matrix) == sp.csr_matrix:
        matrix = matrix.T
    elif not type(matrix) == sp.csc_matrix:
        raise TypeError(f"Expected a sparse matrix of type csr_matrix or csc_matrix, got {type(matrix)}")
    
    data    = matrix.data
    indices = matrix.indices.astype(np.int32)
    indptr  = matrix.indptr.astype(np.int32)
    shape   = np.array(matrix.shape, dtype=np.int64)

    barcodes = metadata['soma_joinid'].astype(str).tolist()
    gene_ids   = gene_metadata['feature_id'].astype(str).tolist()
    gene_names = gene_metadata['feature_name'].astype(str).tolist()

    os.makedirs(save_dir, exist_ok=True)

    metadata.to_csv(os.path.join(save_dir, f"{filename}.csv"), index=False)

    with h5py.File(os.path.join(save_dir, f"{filename}.h5"), "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("data",    data=data)
        grp.create_dataset("indices", data=indices)
        grp.create_dataset("indptr",  data=indptr)
        grp.create_dataset("shape",   data=shape)
        dt = h5py.special_dtype(vlen=str)
        grp.create_dataset("barcodes", data=np.array(barcodes, dtype=object), dtype=dt)
        feat = grp.create_group("features")
        feat.create_dataset("id",   data=np.array(gene_ids,   dtype=object), dtype=dt)
        feat.create_dataset("name", data=np.array(gene_names, dtype=object), dtype=dt)
        feat.create_dataset("feature_type", data=np.array(["Gene Expression"] * len(gene_ids), dtype=object), dtype=dt)