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


def setup_datapipes(data_dir: str, human_masks: str, mouse_masks: str):
    human_pipe = SpeciesDataPipe(
        directory_path=data_dir,
        npz_masks=[f"{human_mask}.npz" for human_mask in human_masks],
        metadata_masks=[f"{human_mask}.pkl" for human_mask in human_masks],
        batch_size=1000,
        allow_partials=True,
        shuffle=False,
        return_dense=True,
        verbose=True,
    )
    mouse_pipe = SpeciesDataPipe(
        directory_path=data_dir,
        npz_masks=[f"{mouse_mask}.npz" for mouse_mask in mouse_masks],
        metadata_masks=[f"{mouse_mask}.pkl" for mouse_mask in mouse_masks],
        batch_size=1000,
        allow_partials=True,
        shuffle=False,
        return_dense=True,
        verbose=True,
    )
    return human_pipe, mouse_pipe


def setup_dataloaders(data_dir: str):

    gids = np.random.choice(
        np.arange(1, 158), size=16, replace=False
    )
    
    human_masks = [f"human*_{gid}" for gid in gids]
    mouse_masks = [f"mouse*_{gid}" for gid in gids]

    human_pipe, mouse_pipe = setup_datapipes(data_dir, human_masks, mouse_masks)

    human_dataloader = DataLoader(
        dataset=human_pipe,
        batch_size=None,
        shuffle=False,
        collate_fn=lambda x: x,
        persistent_workers=False,
        num_workers=2,
    )
    mouse_dataloader = DataLoader(
        dataset=mouse_pipe,
        batch_size=None,
        shuffle=False,
        collate_fn=lambda x: x,
        persistent_workers=False,
        num_workers=2,
    )

    return human_dataloader, mouse_dataloader


def convert_to_tensor(batch: sp.csr_matrix):
    return torch.sparse_csr_tensor(
        crow_indices=batch.indptr,
        col_indices=batch.indices,
        values=batch.data,
        size=batch.shape,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )

def get_correlations(model: CMMVAEModel, data_dir: str, save_dir: str):
    human_dataloader, mouse_dataloader = setup_dataloaders(data_dir)

    for (human_batch, human_metadata), (mouse_batch, mouse_metadata) in zip(
        human_dataloader, mouse_dataloader
    ):
        human_batch = human_batch.cuda()
        mouse_batch = mouse_batch.cuda()

        model.module.eval()
        with torch.no_grad():
            _, _, _, human_out, _ = model.module(
                human_batch, human_metadata, RK.HUMAN, cross_generate=True
            )
            _, _, _, mouse_out, _ = model.module(
                mouse_batch, mouse_metadata, RK.MOUSE, cross_generate=True
            )
        save_data = convert_to_csr(human_out, mouse_out)
        metadata = {RK.HUMAN: human_metadata, RK.MOUSE: mouse_metadata}
        save_correlations(save_data, metadata, save_dir, gid=human_metadata["group_id"].iloc[0])

def convert_to_csr(
    human_xhats: dict[str: torch.Tensor],
    mouse_xhats: dict[str: torch.Tensor],
):
    converted = {}
    for output_species, xhat in human_xhats.items():
        nparray = xhat.cpu().numpy()
        converted[f"human_to_{output_species}"] = sp.csr_matrix(nparray)
    for output_species, xhat in mouse_xhats.items():
        nparray = xhat.cpu().numpy()
        converted[f"mouse_to_{output_species}"] = sp.csr_matrix(nparray)
    return converted
    

def save_correlations(data: dict[str, sp.csr_matrix], metadata: pd.DataFrame, save_dir: str, gid: int):
    for file_name, data in data.items():
        sp.save_npz(
            os.path.join(save_dir, f"{file_name}_{gid}.npz"),
            data
        )
    for species, md in metadata.items():
        md.to_pickle(
            os.path.join(save_dir, f"{species}_metadata_{gid}.pkl")
        )

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
    help="Directory where filtered correlation data is saved",
)
@click.option(
    "--save_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory where correlation outputs are saved",
)
@click.pass_context
def correlations(ctx: click.Context, correlation_data: str, save_dir: str):
    """Run using the LightningCli."""
    if ctx.args:
        # Ensure `args` is passed as the command-line arguments
        sys.argv = [sys.argv[0]] + ctx.args

    cli = CMMVAECli(run=False)
    model = type(cli.model).load_from_checkpoint(
        cli.config["ckpt_path"], module=cli.model.module
    )
    get_correlations(model, correlation_data, save_dir)


if __name__ == "__main__":
    correlations()
