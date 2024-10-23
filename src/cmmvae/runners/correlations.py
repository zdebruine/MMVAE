"""
Get R^2 correlations between cis and cross species generations
"""
import os
import sys
import click
import torch

import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch.utils.data import DataLoader

from cmmvae.models import CMMVAEModel
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cli import CMMVAECli
from cmmvae.data.local import SpeciesDataPipe

FILTERED_BY_CATEGORIES = ["assay", "cell_type", "tissue", "sex"]

def setup_datapipes(data_dir: str):
    human_pipe = SpeciesDataPipe(
        directory_path= data_dir,
        npz_masks= "human*.npz",
        metadata_masks= "human*.pkl",
        batch_size= 10000,
        allow_partials=True,
        shuffle= False,
        return_dense= False,
        verbose= True,
    )
    mouse_pipe = SpeciesDataPipe(
        directory_path= data_dir,
        npz_masks= "mouse*.npz",
        metadata_masks= "mouse*.pkl",
        batch_size= 10000,
        allow_partials=True,
        shuffle= False,
        return_dense= False,
        verbose= True,
    )
    return human_pipe, mouse_pipe

def setup_dataloaders(data_dir: str):
    human_pipe, mouse_pipe = setup_datapipes(data_dir)
    
    human_dataloader = DataLoader(
                        dataset=human_pipe,
                        batch_size=None,
                        shuffle=False,
                        collate_fn=lambda x: x,
                        persistent_workers=False,
                        num_workers=6,
                    )
    mouse_dataloader = DataLoader(
                        dataset=mouse_pipe,
                        batch_size=None,
                        shuffle=False,
                        collate_fn=lambda x: x,
                        persistent_workers=False,
                        num_workers=6,
                    )
    
    return human_dataloader, mouse_dataloader

def convert_to_tensor(batch: sp.csr_matrix):
    return torch.sparse_csr_tensor(
        crow_indices=batch.indptr,
        col_indices=batch.indices,
        values=batch.data,
        size=batch.shape,
    )

def calc_correlations(human_out: np.ndarray, mouse_out: np.ndarray, n_samples: int):
    with np.errstate(divide="ignore", invalid="ignore"):
                human_correlations = np.corrcoef(human_out)
                mouse_correlations = np.corrcoef(mouse_out)

                human_cis = np.round(np.nan_to_num(human_correlations[:n_samples, :n_samples]).mean(), 3)
                human_cross = np.round(np.nan_to_num(human_correlations[n_samples:, n_samples:]).mean(), 3)
                human_comb = np.round(np.nan_to_num(human_correlations[:n_samples, n_samples:]).mean(), 3)

                mouse_cis = np.round(np.nan_to_num(mouse_correlations[:n_samples, :n_samples]).mean(), 3)
                mouse_cross = np.round(np.nan_to_num(mouse_correlations[n_samples:, n_samples:]).mean(), 3)
                mouse_comb = np.round(np.nan_to_num(mouse_correlations[:n_samples, n_samples:]).mean(), 3)
    
    return pd.DataFrame(
        {
            "human_cis": [human_cis],
            "human_cross": [human_cross],
            "human_comb": [human_comb],
            "mouse_cis": [mouse_cis],
            "mouse_cross": [mouse_cross],
            "mouse_comb": [mouse_comb]
        }
    )

def get_correlations(model: CMMVAEModel, data_dir: str):
    
    human_dataloader, mouse_dataloader = setup_dataloaders(data_dir)
    
    correlations = pd.DataFrame(
         columns=[
              "tag", "group_id", "num_samples",
              "human_cis", "human_cross", "human_comb",
              "mouse_cis", "mouse_cross", "mouse_comb"
            ]
    )

    for (human_batch, human_metadata), (mouse_batch, mouse_metadata) in zip(human_dataloader, mouse_dataloader):

        n_samples = human_metadata["num_samples"].iloc[0]

        model.module.eval()
        with torch.no_grad():
            _, _, _, human_out, _ = model.module(human_batch, human_metadata, RK.HUMAN, cross_generate=True)
            _, _, _, mouse_out, _ = model.module(mouse_batch, mouse_metadata, RK.MOUSE, cross_generate=True)

        human_stacked_out = np.vstack((human_out[RK.HUMAN].numpy(), mouse_out[RK.HUMAN].numpy()))
        mouse_stacked_out = np.vstack((mouse_out[RK.MOUSE].numpy(), human_out[RK.MOUSE].numpy()))

        avg_correlations = calc_correlations(human_stacked_out, mouse_stacked_out, n_samples)

        avg_correlations["tag"] = " ".join([human_metadata[cat].iloc[0] for cat in FILTERED_BY_CATEGORIES])
        avg_correlations["group_id"] = human_metadata["group_id"].iloc[0]
        avg_correlations["num_samples"] = n_samples

    correlations = pd.concat([correlations, avg_correlations], ignore_index=True)

    return correlations

def save_correlations(correlations: pd.DataFrame, save_dir: str):
     correlations = correlations.sort_values("group_id")
     correlations.to_csv(os.path.join(save_dir, "correlations.csv"), index=False)
     correlations.to_pickle(os.path.join(save_dir, "correlations.pkl"))

@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--correlation_data",
    type=click.Path(exists=True),
    required=True,
    help="Directory where filtered correlation data is saved"
)
@click.option(
    "--save_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory where correlation outputs are saved"
)
@click.pass_context
def correlations(ctx: click.Context, correlation_data: str, save_dir: str):
    """Run using the LightningCli."""
    if ctx.args:
        # Ensure `args` is passed as the command-line arguments
        sys.argv = [sys.argv[0]] + ctx.args

    cli = CMMVAECli(run=False)
    correlations = get_correlations(cli.model, correlation_data)
    save_correlations(correlations, save_dir)

if __name__ == "__main__":
    correlations()