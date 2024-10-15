import os
import sys
from typing import Iterable, Optional

import click
import pandas as pd
import csv

from cmmvae.data.local import SpeciesDataModule
from cmmvae.runners.cli import CMMVAECli, context_settings


def get_metadata_files(datamodule: SpeciesDataModule):
    return {
        species.name: [
            os.path.join(species.directory_path, file)
            for file in [
                *species.train_metadata_masks,
                *species.val_metadata_masks,
                *species.test_metadata_masks,
            ]
        ]
        for species in datamodule.species
    }


def accumulate_species_dataframes(pickle_files: dict[str, list[str]]):
    """Accumlates dataframes by appending 'species'"""
    dataframes = []

    # Loop through each pickle file, load the DataFrame, and accumulate them
    for name, files in pickle_files.items():
        for file in files:
            try:
                file = os.path.join(file)
                df = pd.read_pickle(file)
                df["species"] = name
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    # Concatenate all DataFrames
    if dataframes:
        accumulated_df = pd.concat(dataframes, ignore_index=True)
    else:
        accumulated_df = pd.DataFrame()

    return accumulated_df


def differentiate_expression(
    df: pd.DataFrame, shared_labels: Optional[Iterable] = None
):
    # get all labels from the dataset
    labels = set(df.columns)
    # initialize all shared_labels as a set
    shared_labels = set(shared_labels) if shared_labels else set()

    if shared_labels:
        # get all labels in dataset that are in shared_labels
        known_labels = labels & shared_labels
        try:
            assert len(known_labels) == len(shared_labels)
        except AssertionError:
            unknown_labels = shared_labels.difference(known_labels)
            raise ValueError(f"Found unknown labels: {unknown_labels}")
        # remove all shared_labels from labels
        labels.difference_update(shared_labels)
    return labels, shared_labels


def write_lines_to_file(
    lines: Iterable[str],
    root_dir: str,
    label: str,
    makefilename=lambda label: f"unique_expression_{label}.csv",
    assert_nonexisting_or_empty: bool = False,
):
    file_name = makefilename(label)
    file_path = os.path.join(root_dir, file_name)

    if assert_nonexisting_or_empty:
        assert not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    os.makedirs(root_dir, exist_ok=True)
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)

        for item in lines:
            writer.writerow([item])


def write_unique_expressions(root_dir: str, column: str, df: pd.DataFrame):
    unique_values = df[column].dropna().unique()
    write_lines_to_file(lines=unique_values, root_dir=root_dir, label=column)


def record_expression(
    pickle_files: dict[str, list[str]],
    root_dir: str,
    shared_labels: Optional[Iterable[str]] = None,
):
    if os.path.exists(root_dir):
        print("Expressions already generated as directory exists!")
        return

    df = accumulate_species_dataframes(pickle_files)
    labels, shared_labels = differentiate_expression(df, shared_labels=shared_labels)
    species_names = list(pickle_files.keys())
    shared_dir = os.path.join(root_dir, "shared")

    for column in shared_labels:
        write_unique_expressions(shared_dir, column, df)

    for column in labels:
        for species in species_names:
            species_dir = os.path.join(root_dir, species)
            species_df = df[df["species"] == species]
            write_unique_expressions(species_dir, column, species_df)


@click.command(context_settings=context_settings())
@click.pass_context
def expression(ctx: click.Context):
    sys.argv = [sys.argv[0]]
    # Example of further processing with CMMVAECli
    cli = CMMVAECli(args=ctx.args, only_data=True, run=False)

    datamodule = cli.datamodule

    pickle_files = get_metadata_files(datamodule)

    record_expression(
        pickle_files=pickle_files,
        root_dir=datamodule.conditionals_directory,
        shared_labels=datamodule.shared_conditionals,
    )


if __name__ == "__main__":
    expression()
