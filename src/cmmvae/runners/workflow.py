import click

from cmmvae.runners.expression import expression
from cmmvae.runners.cli import cli
from cmmvae.runners.umap_predictions import umap_predictions
from cmmvae.runners.merge_predictions import merge_predictions


@click.group()
def workflow():
    """Workflow commands for experiments."""


workflow.add_command(expression)
workflow.add_command(cli)
workflow.add_command(umap_predictions)
workflow.add_command(merge_predictions)
