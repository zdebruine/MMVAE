from cmmvae.models import CMMVAEModel
import cmmvae.data.local.cellxgene_datapipe as cellxgene_datapipe
import cmmvae.data.local.multi_modal_loader as multi_modal_loader

import torch
import torch.nn as nn

import click
import os

from torch.utils.tensorboard import SummaryWriter

import yaml


def create_discriminators(latent_dim):
    """
    Create discriminators for the latent space and the two species.

    Args:
        latent_dim (int): Dimension of the latent space of the CMMVAE model.
    """

    latent_disc = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    human_disc = nn.Sequential(
        nn.Linear(52417, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    mouse_disc = nn.Sequential(
        nn.Linear(60664, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    discriminators = nn.ModuleDict(
        {"latent": latent_disc, "human": human_disc, "mouse": mouse_disc}
    )

    return discriminators


# Functions for annotating data with species tag
def h_generator(source):
    tensor, metadata = source
    return tensor, metadata, "human"


def m_generator(source):
    tensor, metadata = source
    return tensor, metadata, "mouse"


def create_dataloader(batch_size: int):
    human_dataloader = cellxgene_datapipe.SpeciesDataPipe(
        "/mnt/projects/debruinz_project/summer_census_data/3m_subset",
        "3m_human_counts_14.npz",
        "3m_human_metadata_14.pkl",
        batch_size,
        transform_fn=h_generator,
    )
    mouse_dataloader = cellxgene_datapipe.SpeciesDataPipe(
        "/mnt/projects/debruinz_project/summer_census_data/3m_subset",
        "3m_mouse_counts_14.npz",
        "3m_mouse_metadata_14.pkl",
        batch_size,
        transform_fn=m_generator,
    )

    mm_loader = multi_modal_loader.MultiModalDataLoader(
        human_dataloader, mouse_dataloader
    )

    return mm_loader


def train_md(
    model: CMMVAEModel,
    dataloader,
    discriminators: dict,
    writer: SummaryWriter,
    device,
    num_epochs=15,
    lr=0.001,
):
    d = discriminators
    optimzers = {
        "latent": torch.optim.Adam(d["latent"].parameters(), lr=lr),
        "human": torch.optim.Adam(d["human"].parameters(), lr=lr),
        "mouse": torch.optim.Adam(d["mouse"].parameters(), lr=lr),
    }

    for disc in d.values():
        disc.to(device)
        disc.train()

    model.to(device)
    model.eval()
    for epoch in range(num_epochs):
        latent_loss, human_loss, mouse_loss = 0, 0, 0
        batch_iteration = 0
        for x, metadata, expert_id in dataloader:
            batch_iteration += 1
            x = x.to(device)

            with torch.no_grad():
                qz, pz, z, xhats, cg_xhats, _ = model.module(
                    x, metadata, expert_id, cross_generate=True
                )
            if x.layout == torch.sparse_csr:
                x = x.to_dense()

            for disc in d.values():
                disc.zero_grad()

            human_label = torch.zeros(model.hparams.batch_size, 1, device=device)
            mouse_label = torch.ones(model.hparams.batch_size, 1, device=device)
            truth = human_label if expert_id == "human" else mouse_label

            latent_disc_output = d["latent"](z)

            l_loss = nn.functional.binary_cross_entropy(
                latent_disc_output, truth, reduction="mean"
            )

            l_loss.backward()
            optimzers["latent"].step()

            latent_loss += l_loss
            if expert_id == "human":
                species_disc_output = d["mouse"](cg_xhats["mouse"])
                h_loss = nn.functional.binary_cross_entropy(
                    species_disc_output, human_label, reduction="mean"
                )
                human_loss += h_loss

                h_loss.backward()
                optimzers["mouse"].step()

            else:
                species_disc_output = d["human"](cg_xhats["human"])
                m_loss = nn.functional.binary_cross_entropy(
                    species_disc_output, mouse_label, reduction="mean"
                )
                m_loss.backward()

                mouse_loss += m_loss
                optimzers["human"].step()

        writer.add_scalar("meta_disc/md_latent", latent_loss / batch_iteration, epoch)
        writer.add_scalar("meta_disc/md_human", human_loss / batch_iteration, epoch)
        writer.add_scalar("meta_disc/md_mouse", mouse_loss / batch_iteration, epoch)


def run_md_tests(log_directory, checkpoint_path, config_path):
    """
    Perform setup for training the meta discriminators. Start training the discriminators.

    Args:
        log_directory (str): Path to the directory where tensorboard logs are stored.
        checkpoint_path (str): Path to the checkpoint of the CMMVAE model.
        config_path (str): Path to the yaml config of the CMMVAE model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_init_args = config["model"]["init_args"]
    batch_size = config["data"]["init_args"]["species"][0]["init_args"]["batch_size"]
    latent_dim = config["model"]["init_args"]["module"]["init_args"]["vae"][
        "init_args"
    ]["latent_dim"]

    model = CMMVAEModel.load_from_checkpoint(checkpoint_path, **model_init_args)
    data_loader = create_dataloader(batch_size)
    discrimintors = create_discriminators(latent_dim)
    writer = SummaryWriter(log_directory)

    train_md(model, data_loader, discrimintors, writer, device)


@click.command()
@click.option(
    "--log_dir",
    type=click.Path(exists=True),
    required=True,
    default=lambda: os.environ.get("DIRECTORY", ""),
    help="Directory where meta discriminator tensorboard logs are stored",
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True),
    required=True,
    help="CMMVAE model checkpoint to be used",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="yaml config of CMMVAE model",
)
def meta_discriminator(log_dir, ckpt, config):
    run_md_tests(log_dir, ckpt, config)


if __name__ == "__main__":
    pass
    # meta_discriminator()
