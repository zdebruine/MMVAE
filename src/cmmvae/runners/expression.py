import os
import sys

import click
import pandas as pd
import csv

from cmmvae.runners.cli import CMMVAECli
from cmmvae.data.local import SpeciesDataModule


def gather_unique_metadata(dfs, keys=None):
    """
    Gathers unique metadata from a list of DataFrames based on the specified columns (keys).
    If no keys are specified, all columns will be considered for each DataFrame.

    Parameters:
    dfs (list): A list of pandas DataFrames.
    keys (list or None): A list of column names to consider, or None to use all columns.

    Returns:
    dict: A dictionary where the keys are column names and the values are sets of unique metadata for that column.
    """
    metadata_dict = {}

    for df in dfs:
        # If no specific keys are provided, use all columns in the dataframe
        columns_to_process = keys if keys else df.columns

        for column in columns_to_process:
            if column in df.columns:
                # Initialize the set if this column is not yet in the dictionary
                if column not in metadata_dict:
                    metadata_dict[column] = set()

                # Add unique values from the current dataframe to the set
                unique_values = df[column].dropna().unique()
                metadata_dict[column].update(unique_values)

    return metadata_dict


def record_expression(
    datamodule: SpeciesDataModule,
    batch_keys: list[str],
    log_dir="",
):
    dfs = []
    for species in datamodule.species:
        for file in [
            *species.train_metadata_masks,
            *species.val_metadata_masks,
            *species.test_metadata_masks,
        ]:
            dfs.append(pd.read_pickle(file))

    for key, unique in gather_unique_metadata(dfs).items():
        with open(
            os.path.join(log_dir, f"unique_expression_{key}.csv", "w", newline="")
        ) as file:
            writer = csv.writer(file)

            # Write each set element as a new row in the CSV
            for item in unique:
                writer.writerow([item])


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.pass_context
def expression(ctx: click.Context):
    """Run using the LightningCli."""
    if ctx.args:
        # Ensure `args` is passed as the command-line arguments
        sys.argv = [sys.argv[0]] + ctx.args

    cli = CMMVAECli(run=False)
    record_expression(
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    expression()
