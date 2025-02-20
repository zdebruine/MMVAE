"""
Get R^2 correlations between cis and cross species generations
"""
import os
import sys
import click
import torch

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from torchmetrics.regression import PearsonCorrCoef

from cmmvae.models import CMMVAEModel
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cli import CMMVAECli
from cmmvae.runners.cross_generation import CrossGenerator

FILE_PATTERN = 'human_filtered_'
EPSILON = 1e-4

# def r_squared(tensor: torch.Tensor):
#     # print("ON PRE_TRANSPOSE", flush=True)
#     # print(torch.cuda.memory_summary(), flush=True)
#     tensor = tensor + 1e-3 #Epsilon
#     print("Plus Epsilon", flush=True)
#     print(tensor, flush=True)
#     pearson_correlations = torch.corrcoef(tensor.t())
#     print("Correlations root", flush=True)
#     print(pearson_correlations, flush=True)
#     # print("ON POST TRANSPOSE", flush=True)
#     # print(torch.cuda.memory_summary(), flush=True)
#     pearson_correlations = pearson_correlations.pow(2)
#     print("Correlations squared", flush=True)
#     print(pearson_correlations, flush=True)

#     # Remove bias in the output due to each sample being perfectly correlated with itself
#     off_diag_sum = pearson_correlations.sum() - torch.diag(pearson_correlations).sum()
#     n_samples = pearson_correlations.size(0)
#     num_off_diag = n_samples * (n_samples - 1)
#     avg_corr_off_diag = off_diag_sum / num_off_diag

#     r2 = round(avg_corr_off_diag.item(), 3)
#     return r2

def get_r_squared(
    cross_generator: CrossGenerator,
    context_data_dir: str,
):
    correlations = pd.DataFrame(
        columns=[
            'primary_context_id',
            'secondary_context_id',
            'primary_cis',
            'primary_cross',
            'primary_comb',
            'primary_rel',
            'secondary_cis',
            'secondary_cross',
            'secondary_comb',
            'secondary_rel',
            'mod_type',
            'mod_value',
        ] + RK.FILTER_CATEGORIES
    )
    pearson_correlation = PearsonCorrCoef(num_outputs=60530)
    pearson_correlation.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    context_references = os.path.join(context_data_dir, "group_references.csv")
    contexts = cross_generator.get_contexts(context_references)

    i = 0
    for primary_context, secondary_contexts in contexts.items():
        i += 1
        # Limit the number of samples processes while testing functionality
        if i > 10:
            break
        # Transpose the outputs as the Pearson CorrCoef expects
        out = cross_generator.cross_generate(context_data_dir, primary_context, secondary_contexts, transpose= False)
        print(out)
        primary_cis_xhat = out[f"{primary_context}_to_{primary_context}"] + EPSILON
        print(pearson_correlation(primary_cis_xhat, primary_cis_xhat))
        print(pearson_correlation(primary_cis_xhat, primary_cis_xhat).mean().item())
        primary_cis_r2 = round(pearson_correlation(primary_cis_xhat, primary_cis_xhat).mean().item(), 3)
        for secondary_context, modifications in secondary_contexts.items():
            context_correlations = pd.DataFrame(
                columns=[
                    "primary_context_id",
                    "secondary_context_id",
                    "primary_cis",
                    "primary_cross",
                    "primary_comb",
                    "primary_rel",
                    "secondary_cis",
                    "secondary_cross",
                    "secondary_comb",
                    "secondary_rel",
                    "mod_type",
                    "mod_value",
                ] + RK.FILTER_CATEGORIES
            )

            secondary_cis_xhat = out[f"{secondary_context}_to_{secondary_context}"] + EPSILON
            secondary_cross_xhat = out[f"{primary_context}_to_{secondary_context}"] + EPSILON
            primary_cross_xhat = out[f"{secondary_context}_to_{primary_context}"] + EPSILON

            primary_cross_r2 = round(pearson_correlation(secondary_cross_xhat, secondary_cross_xhat).mean().item(), 3)
            primary_combined_r2 = round(pearson_correlation(secondary_cross_xhat, primary_cis_xhat).mean().item(), 3)

            secondary_cis_r2 = round(pearson_correlation(secondary_cis_xhat, secondary_cis_xhat).mean().item(), 3)
            secondary_cross_r2 = round(pearson_correlation(primary_cross_xhat, primary_cross_xhat).mean().item(), 3)
            secondary_combined_r2 = round(pearson_correlation(primary_cross_xhat, secondary_cis_xhat).mean().item(), 3)


            primary_relative_r2 = round((2 * primary_combined_r2) / (primary_cis_r2 + primary_cross_r2), 3)
            secondary_relative_r2 = round((2 * secondary_combined_r2) / (secondary_cis_r2 + secondary_cross_r2), 3)

            context_correlations["primary_context_id"] = [primary_context]
            context_correlations["secondary_context_id"] = [secondary_context]

            context_correlations["primary_cis"] = [primary_cis_r2]
            context_correlations["primary_cross"] = [primary_cross_r2]
            context_correlations["primary_comb"] = [primary_combined_r2]
            context_correlations["primary_rel"] = [primary_relative_r2]
            context_correlations["secondary_cis"] = [secondary_cis_r2]
            context_correlations["secondary_cross"] = [secondary_cross_r2]
            context_correlations["secondary_comb"] = [secondary_combined_r2]
            context_correlations["secondary_rel"] = [secondary_relative_r2]

            context_correlations["mod_type"] = modifications["mod_type"]
            context_correlations["mod_value"] = modifications["mod_value"]

            for cat in RK.FILTER_CATEGORIES:
                context_correlations[cat] = out["metadata"][cat].iloc[0]
            
            correlations = pd.concat([correlations, context_correlations], ignore_index=True)

    return correlations

def save_correlations(correlations: pd.DataFrame, save_dir: str):
    correlations = correlations.sort_values("primary_context_id")
    correlations.to_csv(os.path.join(save_dir, "correlations_test.csv"), index=False)

@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--model_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory where the model config and ckpt are stored",
)
@click.option(
    "--context_data_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory where the filtered context data is stored",
)
@click.option(
    "--save_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory where the correlations are saved",
)
@click.pass_context
def correlations(ctx: click.Context, model_dir: str, context_data_dir: str, save_dir: str):
    """Run using the LightningCli."""
    
    cross_gen = CrossGenerator(model_dir)
    correlations_df = get_r_squared(cross_gen, context_data_dir)
    save_correlations(correlations_df, save_dir)


if __name__ == "__main__":
    correlations()
