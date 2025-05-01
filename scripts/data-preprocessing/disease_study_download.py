import argparse as ap
import cellxgene_census as cxg
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import tiledbsoma as tdb
import cellxgene_ontology_guide.ontology_parser as op

from data_processing_functions import normalize_data, save_data_to_disk, verify_data

CENSUS_VERSION = "2024-07-01"
VALUE_FILTER = "is_primary_data == True and suspension_type != 'unknown' and cell_type != 'unknown' and tissue_general != 'unknown' and sex != 'unknown' and development_stage_ontology_term_id not in ['unknown', 'HsapDv:0000087', 'HsapDv:0000204'] and assay != 'unknown' and dataset_id != 'unknown' and donor_id != 'unknown'" # Metadata filters applied when retrieving data
NORMAL = "disease == 'normal' and "
COVID = "disease == 'COVID-19' and "
LUNG_ADENOCARCINOMA = "disease == 'lung adenocarcinoma' and "
CROHN = "disease == 'Crohn disease' and "

SOMA_JOINID = 'soma_joinid'
HUMAN = "homo_sapiens"
CHUNK_SIZE = 499968

PARSER = op.OntologyParser()

HUMAN_EMBRYO = "HsapDv:0000002"
HUMAN_FETAL = "HsapDv:0000037"
HUMAN_IMMATURE = "HsapDv:0000264"
HUMAN_YOUNG_ADULT = "HsapDv:0000266"
HUMAN_MIDDLE_ADULT = "HsapDv:0000267"
HUMAN_LATE_ADULT = "HsapDv:0000227"

HUMAN_EMBRYO_DESCENDANTS = PARSER.get_term_descendants(HUMAN_EMBRYO, include_self=True)
HUMAN_FETAL_DESCENDANTS = PARSER.get_term_descendants(HUMAN_FETAL, include_self=True)
HUMAN_IMMATURE_DESCENDANTS = PARSER.get_term_descendants(HUMAN_IMMATURE, include_self=True)
HUMAN_YOUNG_ADULT_DESCENDANTS = PARSER.get_term_descendants(HUMAN_YOUNG_ADULT, include_self=True)
HUMAN_MIDDLE_ADULT_DESCENDANTS = PARSER.get_term_descendants(HUMAN_MIDDLE_ADULT, include_self=True)
HUMAN_LATE_ADULT_DESCENDANTS = PARSER.get_term_descendants(HUMAN_LATE_ADULT, include_self=True)

def tag_human(term_id):
    
    if term_id == "unknown":
        return "unknown"
    elif term_id in HUMAN_EMBRYO_DESCENDANTS:
        return "embryonic"
    elif term_id in HUMAN_FETAL_DESCENDANTS:
        return "fetal"
    elif term_id in HUMAN_IMMATURE_DESCENDANTS:
        return "immature"
    elif term_id in HUMAN_YOUNG_ADULT_DESCENDANTS:
        return "young_adult"
    elif term_id in HUMAN_MIDDLE_ADULT_DESCENDANTS:
        return "middle_adult"
    elif term_id in HUMAN_LATE_ADULT_DESCENDANTS:
        return "late_adult"
    else:
        if PARSER.is_term_deprecated(term_id):
            replacement = PARSER.get_term_replacement(term_id)
            if replacement is not None:
                return tag_human(replacement)
            else:
                consider = PARSER.get_term_metadata(term_id)["consider"]
                if consider is not None:
                    return tag_human(consider[0])
                else:
                    return "unknown"
        return "unknown"


def process_chunk(
    save_dir: os.PathLike,
    ids: list[int],
    chunk_n: int,
    file_tag: str,
):
    """
    Helper function to download and save a chunk of data.

    This is the spawn point of the processes when MultiProcessing
    is used.

    Args:
        save_dir (PathLike):
            Directory to save the data to.
        species (str):
            The name of the species whose data is being downloaded.
            NOTE: The taxonomic name is expected and is later mapped
            to the English name when saving the data.
        ids (list[int]):
            The soma_joinids to query the data that is to be downloaded.
        chunk_n (int):
            Current chunk being processes. Appended to the filename when
            its saved to disk.
    """
    with cxg.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cxg.get_anndata(
            census= census,
            organism= HUMAN,
            measurement_name= "RNA",
            X_name= "raw",
            obs_coords=ids
        )

    normalize_data(adata.X)
    adata.obs["dev_stage"] = adata.obs['development_stage_ontology_term_id'].apply(tag_human)
    permutation = np.random.permutation(range(adata.X.shape[0]))
    save_data_to_disk(
        data_path=os.path.join(save_dir, f'{file_tag}_disease_study_counts_{chunk_n}.npz'),
        data=adata.X[permutation, :],
        metadata_path=os.path.join(save_dir, f'{file_tag}_disease_study_metadata_{chunk_n}.pkl'),
        metdata=adata.obs.iloc[permutation].reset_index(drop=True)
    )

def main():

    np.random.seed(42)

    with cxg.open_soma(census_version=CENSUS_VERSION) as census:

        healthy_data = cxg.get_obs(census, HUMAN, value_filter=NORMAL + VALUE_FILTER, column_names=[SOMA_JOINID])
        covid_data = cxg.get_obs(census, HUMAN, value_filter=COVID + VALUE_FILTER, column_names=[SOMA_JOINID])
        lung_adenocarcinoma_data = cxg.get_obs(census, HUMAN, value_filter=LUNG_ADENOCARCINOMA + VALUE_FILTER, column_names=[SOMA_JOINID])
        crohn_data = cxg.get_obs(census, HUMAN, value_filter=CROHN + VALUE_FILTER, column_names=[SOMA_JOINID])

    healthy_soma_ids = set(healthy_data[SOMA_JOINID])
    covid_soma_ids = set(covid_data[SOMA_JOINID])
    lung_adenocarcinoma_soma_ids = set(lung_adenocarcinoma_data[SOMA_JOINID])
    crohn_soma_ids = set(crohn_data[SOMA_JOINID])

    num_covid = len(covid_soma_ids)
    num_lung_adenocarcinoma = len(lung_adenocarcinoma_soma_ids)
    num_crohn = len(crohn_soma_ids)

    covid_cell_types = covid_data['cell_type'].unique().tolist()
    covid_cells = np.random.choice(covid_cell_types, len(covid_cell_types) // 2, replace=False)
    filtered_percent = ((num_covid - covid_data['cell_type'].isin(covid_cells).sum()) / num_covid) * 100
    while filtered_percent > 75 or filtered_percent < 50:
        covid_cells = np.random.choice(covid_cell_types, len(covid_cell_types) // 2, replace=False)
        filtered_percent = ((num_covid - covid_data['cell_type'].isin(covid_cells).sum()) / num_covid) * 100

    lung_adenocarcinoma_cell_types = lung_adenocarcinoma_data['cell_type'].unique().tolist()
    lung_adenocarcinoma_cells = np.random.choice(lung_adenocarcinoma_cell_types, len(lung_adenocarcinoma_cell_types) // 2, replace=False)
    filtered_percent = ((num_lung_adenocarcinoma - lung_adenocarcinoma_data['cell_type'].isin(lung_adenocarcinoma_cells).sum()) / num_lung_adenocarcinoma) * 100
    while filtered_percent > 75 or filtered_percent < 50:
        lung_adenocarcinoma_cells = np.random.choice(lung_adenocarcinoma_cell_types, len(lung_adenocarcinoma_cell_types) // 2, replace=False)
        filtered_percent = ((num_lung_adenocarcinoma - lung_adenocarcinoma_data['cell_type'].isin(lung_adenocarcinoma_cells).sum()) / num_lung_adenocarcinoma) * 100

    crohn_cell_types = crohn_data['cell_type'].unique().tolist()
    crohn_cells = np.random.choice(crohn_cell_types, len(crohn_cell_types) // 2, replace=False)
    filtered_percent = ((num_crohn - crohn_data['cell_type'].isin(crohn_cells).sum()) / num_crohn) * 100
    while filtered_percent > 75 or filtered_percent < 50:
        crohn_cells = np.random.choice(crohn_cell_types, len(crohn_cell_types) // 2, replace=False)
        filtered_percent = ((num_crohn - crohn_data['cell_type'].isin(crohn_cells).sum()) / num_crohn) * 100  

    selected_covid = set(covid_data.loc[covid_data['cell_type'].isin(covid_cells), SOMA_JOINID])
    selected_lung_adenocarcinoma = set(lung_adenocarcinoma_data.loc[lung_adenocarcinoma_data['cell_type'].isin(lung_adenocarcinoma_cells), SOMA_JOINID])
    selected_crohn = set(crohn_data.loc[crohn_data['cell_type'].isin(crohn_cells), SOMA_JOINID])

    filtered_covid = covid_soma_ids.difference(selected_covid)
    filtered_lung_adenocarcinoma = lung_adenocarcinoma_soma_ids.difference(selected_lung_adenocarcinoma)
    filtered_crohn = crohn_soma_ids.difference(selected_crohn)

    full_soma_ids = list(healthy_soma_ids | covid_soma_ids | lung_adenocarcinoma_soma_ids | crohn_soma_ids)
    filtered_soma_ids = list(healthy_soma_ids | filtered_covid | filtered_lung_adenocarcinoma | filtered_crohn)

    np.random.shuffle(full_soma_ids)
    np.random.shuffle(filtered_soma_ids)

    mp.set_start_method('spawn')

    full_directory = "/mnt/projects/debruinz_project/disease_study/full/"
    filtered_directory = "/mnt/projects/debruinz_project/disease_study/filtered/"
    selected_directory = "/mnt/projects/debruinz_project/disease_study/selected/"

    with mp.Pool(processes= 6) as pool:
        chunk_count = 0
        for i in range(0, len(full_soma_ids), CHUNK_SIZE):
            chunk_count += 1
            pool.apply_async(
                func= process_chunk,
                args=(full_directory, full_soma_ids[i:i+CHUNK_SIZE], chunk_count, "full")
            )
        pool.close()
        pool.join()

    with mp.Pool(processes= 6) as pool:
        chunk_count = 0
        for i in range(0, len(filtered_soma_ids), CHUNK_SIZE):
            chunk_count += 1
            pool.apply_async(
                func= process_chunk,
                args=(filtered_directory, filtered_soma_ids[i:i+CHUNK_SIZE], chunk_count, "filtered")
            )
        pool.close()
        pool.join()

    process_chunk(selected_directory, list(selected_covid), 1, "selected_covid")
    process_chunk(selected_directory, list(selected_lung_adenocarcinoma), 1, "selected_lung_adenocarcinoma")
    process_chunk(selected_directory, list(selected_crohn), 1, "selected_crohn")
    
if __name__ == "__main__":
    main()