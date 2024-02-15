"""
2/7/2024 - Anthony Boos
The 3m Human Sampler takes 30k random cells from the full human dataset and produces a 3million cell subset
"""

import random, csv
import scipy.sparse as sp

def load_data(n):
    print(f"Loading chunk {n}")
    with open(f"/active/debruinz_project/human_data/python_data/chunk{n}_metadata.csv", "r") as metadata_file:
        metadata_reader = csv.reader(metadata_file)
        metadata = list(metadata_reader)
    return sp.load_npz(f"/active/debruinz_project/human_data/python_data/human_chunk_{n}.npz"), metadata

def write_data(chunk, metadata, n):
    print("Writing to disk")
    sp.save_npz(f"/active/debruinz_project/CellCensus_3M/3m_human_chunk_{n}.npz", sp.vstack(chunk))
    with open(f"/active/debruinz_project/CellCensus_3M/3m_human_metadata_{n}.csv", "w") as subset_metadata_file:
        metadata_writer = csv.writer(subset_metadata_file)
        metadata_writer.writerows(metadata)
    print(f"Chunk {n} complete!")

def main():
    print("Beginning human sampler")
    n_written = 1 #Track what chunk is to be written
    n_selections = 30000 #Sampling 30K cells per chunk
    max_index = 285341 #Controls the generated range of indicies
    chunk_slices = []
    metadata_slices = []
    for i in range(1,101):
        #Loop and load all 100 chunks of the full dataset
        chunk, metadata = load_data(i)
        if i == 100:
            max_index = 285256 #Chunk 100 is a different size from the rest
        #Get 30k random indicies to slice cells from the current chunk
        indicies = random.sample(range(0, max_index), n_selections)
        for idx in indicies:
            chunk_slices.append(chunk[idx, :])
            metadata_slices.append(metadata[idx+1]) #Adjusting for t he column header row in the metadata
            if len(chunk_slices) == 100000:
                write_data(chunk_slices, metadata_slices, n_written)
                chunk_slices = []
                metadata_slices = []
                n_written += 1

main()
