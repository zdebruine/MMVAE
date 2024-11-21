import os
import re
import sys
import click
import torch
import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp

from collections import defaultdict
from cmmvae.constants import REGISTRY_KEYS as RK

FILTERED_BY_CATEGORIES = ["assay", "cell_type", "tissue", "sex"]

def calc_correlations(human_out: np.ndarray, mouse_out: np.ndarray, n_samples: int):
    with np.errstate(divide="ignore", invalid="ignore"):
        human_correlations = np.corrcoef(human_out)
        mouse_correlations = np.corrcoef(mouse_out)

        human_cis = np.round(
            np.nan_to_num(human_correlations[:n_samples, :n_samples]).mean(), 3
        )
        human_cross = np.round(
            np.nan_to_num(human_correlations[n_samples:, n_samples:]).mean(), 3
        )
        human_comb = np.round(
            np.nan_to_num(human_correlations[:n_samples, n_samples:]).mean(), 3
        )
        human_rel = np.round(
            (2 * human_comb) / (human_cis + human_cross), 3
        )

        mouse_cis = np.round(
            np.nan_to_num(mouse_correlations[:n_samples, :n_samples]).mean(), 3
        )
        mouse_cross = np.round(
            np.nan_to_num(mouse_correlations[n_samples:, n_samples:]).mean(), 3
        )
        mouse_comb = np.round(
            np.nan_to_num(mouse_correlations[:n_samples, n_samples:]).mean(), 3
        )
        mouse_rel = np.round(
            (2 * mouse_comb) / (mouse_cis + mouse_cross), 3
        )

    return pd.DataFrame(
        {
            "human_cis": [human_cis],
            "human_cross": [human_cross],
            "human_comb": [human_comb],
            "human_rel": [human_rel],
            "mouse_cis": [mouse_cis],
            "mouse_cross": [mouse_cross],
            "mouse_comb": [mouse_comb],
            "mouse_rel": [mouse_rel]
        }
    )

def save_correlations(correlations: pd.DataFrame, save_dir: str):
    correlations = correlations.sort_values("group_id")
    correlations.to_csv(os.path.join(save_dir, "correlations.csv"), index=False)
    correlations.to_pickle(os.path.join(save_dir, "correlations.pkl"))

def get_correlations(
        data_files: dict[str: dict[str: sp.csr_matrix]],
        metadata_files: dict[str: dict[str: pd.DataFrame]]
):

    correlations = pd.DataFrame(
        columns=[
            "group_id",
            "num_samples",
            "human_cis",
            "human_cross",
            "human_comb",
            "human_rel",
            "mouse_cis",
            "mouse_cross",
            "mouse_comb",
            "mouse_rel",
            "tag",
        ]
    )
    # print(data_files)
    # print(metadata_files)
    for gid, data in data_files.items():
        # print(gid)
        # print(metadata_files[gid])
        n_samples = metadata_files[gid][RK.HUMAN]["num_samples"].iloc[0]
        human_stacked_out = np.vstack(
            (
                data[f"{RK.HUMAN}_to_{RK.HUMAN}"].toarray(),
                data[f"{RK.MOUSE}_to_{RK.HUMAN}"].toarray()
            )
        )
        mouse_stacked_out = np.vstack(
            (
                data[f"{RK.MOUSE}_to_{RK.MOUSE}"].toarray(),
                data[f"{RK.HUMAN}_to_{RK.MOUSE}"].toarray()
            )
        )

        avg_correlations = calc_correlations(
            human_stacked_out, mouse_stacked_out, n_samples
        )

        avg_correlations["group_id"] = gid
        avg_correlations["num_samples"] = n_samples
        avg_correlations["tag"] = " ".join(
            [metadata_files[gid][RK.HUMAN][cat].iloc[0] for cat in FILTERED_BY_CATEGORIES]
        )

        correlations = pd.concat([correlations, avg_correlations], ignore_index=True)

    return correlations

def correlations(directory: str):
    data_files = defaultdict(dict)
    files = glob.glob(os.path.join(directory, "*.npz"))
    file_pattern = re.compile(r"(.*_to_.*)_(\d+).npz")
    for file in files:
        # print(file)
        match = file_pattern.match(file)
        if match:
            label = match.group(1)
            gid = int(match.group(2))
            label = os.path.basename(label)
            # print(f"label: {label}, GID: {gid}")
            data = sp.load_npz(file)
            data_files[gid][label] = data

    metadata_files = defaultdict(dict)
    files = glob.glob(os.path.join(directory, "*.pkl"))
    file_pattern = re.compile(r"(.*)_metadata_(\d+).pkl")
    for file in files:
        # print(file)
        match = file_pattern.match(file)
        if match:
            species = match.group(1)
            gid = int(match.group(2))
            species = os.path.basename(species)
            # print(f"Species: {species}, GID: {gid}")
            data = pd.read_pickle(file)
            metadata_files[gid][species] = data

    correlations = get_correlations(data_files, metadata_files)
    save_correlations(correlations, directory)

@click.command()
@click.option(
    "--directory",
    type=click.Path(exists=True),
    required=True,
    default=lambda: os.environ.get("DIRECTORY", ""),
    help="Directory where outputs are saved.",
)
def run_correlations(**kwargs):
    correlations(**kwargs)

if __name__ == "__main__":
    run_correlations()