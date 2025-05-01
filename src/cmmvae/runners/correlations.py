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
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional

from torch import Tensor
from typing_extensions import Literal
from torch.nn.functional import mse_loss
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

from cmmvae.models import CMMVAEModel
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.umap_predictions import add_images_to_tensorboard
from cmmvae.runners.cross_generation import CrossGenerator

FILE_PATTERN = 'human_filtered_'
EPSILON = 1e-4

def pairwise_mse(
    x: Tensor,
    y: Optional[Tensor] = None,
) -> Tensor:

    if y is None:
        y = x

    x = x.unsqueeze(1).expand(-1, x.shape[0], -1)
    y = y.unsqueeze(0).expand(y.shape[0], -1, -1)
    mse = mse_loss(x, y, reduction="none")
    summed = mse.sum(dim=-1)

    return summed

def silhouette(
    same,
    nearest,
):
    return (nearest - same) / torch.maximum(same, nearest)

CORRELATION_FUNCTIONS = {"MSE": pairwise_mse, "Euclidean": pairwise_euclidean_distance}#, "Cosine": pairwise_cosine_similarity}

# CORRELATION_FUNCTIONS = {"Euclidean": pairwise_euclidean_distance}

def get_correlations(
    cross_generator: CrossGenerator,
    context_data_dir: str,
    save_dir: str,
):
    files = defaultdict(list)
    # labels = np.array(["A" for _ in range(200)] + ["B" for _ in range(200)])
    for n in range(1, 2):
        contexts = pd.read_pickle(os.path.join(context_data_dir, f"{n}_differences.pkl"))
        correlations = {
            metric_name: pd.DataFrame(
                columns=[
                    RK.CONTEXT_A,
                    RK.CONTEXT_B,
                    # RK.A_TO_A,
                    # RK.A_TO_B,
                    # RK.TO_A,
                    RK.TRUE_A,
                    RK.TRUE_A_AND_A_FROM_A,
                    RK.TRUE_A_AND_A_FROM_B,
                    # RK.RATIO_A,
                    # RK.B_TO_B,
                    # RK.B_TO_A,
                    # RK.TO_B,
                    RK.TRUE_B,
                    RK.TRUE_B_AND_B_FROM_B,
                    RK.TRUE_B_AND_B_FROM_A,
                    # RK.RATIO_B,
                    RK.TRUE_AND_TRUE,
                    RK.RATIO,
                    # RK.TRUE_RATIO,
                    RK.SILHOUETTE_MODEL_A,
                    RK.SILHOUETTE_MODEL_B,
                    RK.SILHOUETTE_MODEL,
                    RK.SILHOUETTE_TRUE_A_FROM_A,
                    RK.SILHOUETTE_TRUE_B_FROM_B,
                    RK.SILHOUETTE_TRUE_CIS,
                    RK.SILHOUETTE_TRUE_A_FROM_B,
                    RK.SILHOUETTE_TRUE_B_FROM_A,
                    RK.SILHOUETTE_TRUE_CROSS,
                    RK.DIFFERENCES,
                    RK.A_DIFFERENCES,
                    RK.B_DIFFERENCES,
                ]
            ) for metric_name in CORRELATION_FUNCTIONS.keys()
        }
        for i in range(len(contexts)):
            context_row = contexts.loc[i]
            out = cross_generator.cross_generate(context_data_dir, context_row, return_real= True)
            
            # context_a_x = out[RK.TRUE_A]
            # context_a_to_a_xhat = out[RK.A_TO_A]
            # context_a_to_b_xhat = out[RK.A_TO_B]
            
            # context_b_x = out[RK.TRUE_B]
            # context_b_to_b_xhat = out[RK.B_TO_B]
            # context_b_to_a_xhat = out[RK.B_TO_A]

            a_out = torch.cat([out[RK.TRUE_A], out[RK.A_TO_A], out[RK.B_TO_A]])
            b_out = torch.cat([out[RK.TRUE_B], out[RK.B_TO_B], out[RK.A_TO_B]])

            # stacked = torch.cat([out[RK.TRUE_A], out[RK.B_TO_A], out[RK.TRUE_B], out[RK.A_TO_B]])

            # km = KMeans(n_clusters=2, random_state=42)
            # labels = km.fit_predict(stacked.cpu().numpy())

            # silhouette = silhouette_samples(stacked.cpu().numpy(), labels)
            # # print(silhouette)
            # silhouette_a = np.mean(silhouette[:200])
            # silhouette_b = np.mean(silhouette[200:])
            
            for metric_name, metric_function in CORRELATION_FUNCTIONS.items():
                metric_correlations = pd.DataFrame(
                    columns=[
                        RK.CONTEXT_A,
                        RK.CONTEXT_B,
                        # RK.A_TO_A,
                        # RK.A_TO_B,
                        # RK.TO_A,
                        RK.TRUE_A,
                        RK.TRUE_A_AND_A_FROM_A,
                        RK.TRUE_A_AND_A_FROM_B,
                        # RK.RATIO_A,
                        # RK.B_TO_B,
                        # RK.B_TO_A,
                        # RK.TO_B,
                        RK.TRUE_B,
                        RK.TRUE_B_AND_B_FROM_B,
                        RK.TRUE_B_AND_B_FROM_A,
                        # RK.RATIO_B,
                        RK.TRUE_AND_TRUE,
                        RK.RATIO,
                        # RK.TRUE_RATIO,
                        RK.SILHOUETTE_MODEL_A,
                        RK.SILHOUETTE_MODEL_B,
                        RK.SILHOUETTE_MODEL,
                        RK.SILHOUETTE_TRUE_A_FROM_A,
                        RK.SILHOUETTE_TRUE_B_FROM_B,
                        RK.SILHOUETTE_TRUE_CIS,
                        RK.SILHOUETTE_TRUE_A_FROM_B,
                        RK.SILHOUETTE_TRUE_B_FROM_A,
                        RK.SILHOUETTE_TRUE_CROSS,
                        RK.DIFFERENCES,
                        RK.A_DIFFERENCES,
                        RK.B_DIFFERENCES,
                    ]
                )

                # a_to_a_tri = torch.tril(metric_function(context_a_to_a_xhat), diagonal=-1)
                # a_to_a = a_to_a_tri.sum() / a_to_a_tri.nonzero().size(0)

                # b_to_a_tri = torch.tril(metric_function(context_b_to_a_xhat), diagonal=-1)
                # b_to_a = b_to_a_tri.sum() / b_to_a_tri.nonzero().size(0)

                # to_a_tri = torch.tril(metric_function(context_a_to_a_xhat, context_b_to_a_xhat))
                # to_a = to_a_tri.sum() / to_a_tri.nonzero().size(0)

                # b_to_b_tri = torch.tril(metric_function(context_b_to_b_xhat), diagonal=-1)
                # b_to_b = b_to_b_tri.sum() / b_to_b_tri.nonzero().size(0)

                # a_to_b_tri = torch.tril(metric_function(context_a_to_b_xhat), diagonal=-1)
                # a_to_b = a_to_b_tri.sum() / a_to_b_tri.nonzero().size(0)

                # to_b_tri = torch.tril(metric_function(context_b_to_b_xhat, context_a_to_b_xhat))
                # to_b = to_b_tri.sum() / to_b_tri.nonzero().size(0)

                # true_a_tri = torch.tril(metric_function(a_out[:100]), diagonal=-1)
                # true_a = true_a_tri.sum() / true_a_tri.nonzero().size(0)

                # true_a_from_a_tri = torch.tril(metric_function(a_out[:100], a_out[100:200]))
                # true_a_from_a = true_a_from_a_tri.sum() / true_a_from_a_tri.nonzero().size(0)

                # true_a_from_b_tri = torch.tril(metric_function(a_out[:100], a_out[200:]))
                # true_a_from_b = true_a_from_b_tri.sum() / true_a_from_b_tri.nonzero().size(0)

                # true_b_tri = torch.tril(metric_function(b_out[:100]), diagonal=-1)
                # true_b = true_b_tri.sum() / true_b_tri.nonzero().size(0)

                # true_b_from_b_tri = torch.tril(metric_function(b_out[:100], b_out[100:200]))
                # true_b_from_b = true_b_from_b_tri.sum() / true_b_from_b_tri.nonzero().size(0)

                # true_b_from_a_tri = torch.tril(metric_function(b_out[:100], b_out[200:]))
                # true_b_from_a = true_b_from_a_tri.sum() / true_b_from_a_tri.nonzero().size(0)

                # true_and_true_tri = torch.tril(metric_function(a_out[:100], b_out[:100]))
                # true_and_true = true_and_true_tri.sum() / true_and_true_tri.nonzero().size(0)
                
                # cis_and_cis_tri = torch.tril(metric_function(context_a_to_a_xhat, context_b_to_b_xhat))
                # cis_and_cis = cis_and_cis_tri.sum() / cis_and_cis_tri.nonzero().size(0)

                # ratio_a = ((2 * to_a) / (a_to_a + b_to_a))
                # ratio_b = ((2 * to_b) / (b_to_b + a_to_b))
                # ratio = (to_a + to_b) / (2 * cis_and_cis)

                # ratio_a = (2 * true_a) / (true_a_from_a + true_a_from_b)
                # ratio_b = (2 * true_b) / (true_b_from_b + true_b_from_a)
                
                # true_ratio = (true_a_from_b + true_b_from_a) / (2 * true_and_true)

                # a_a_dist = metric_function(a_out).sum(dim=1) / (a_out.shape[0] - 1)
                # a_b_dist = metric_function(a_out, b_out).sum(dim=1) / a_out.shape[0]

                # b_b_dist = metric_function(b_out).sum(dim=1) / (b_out.shape[0] - 1)
                # b_a_dist = metric_function(b_out, a_out).sum(dim=1) / b_out.shape[0]

                a_dist = metric_function(a_out) # 300 x 300
                b_dist = metric_function(b_out) # 300 x 300
                a_b_dist = metric_function(a_out, b_out) # 300 x 300


                # torch.set_printoptions(profile="full")
                # print("A: ", a_dist.shape)
                # print("B: ", b_dist.shape)
                # print("AB: ", a_b_dist.shape)

                true_a = a_dist[:100, :100].sum() / (100 * 99) # n * (n-1) to exclude self-compare of 0
                true_a_from_a = a_dist[:100, 100:200].mean()
                true_a_from_b = a_b_dist[:100, 200:].mean()

                true_b = b_dist[:100, :100].sum() / (100 * 99) # n * (n-1) to exclude self-compare of 0
                true_b_from_b = b_dist[:100, 100:200].mean()
                true_b_from_a = a_b_dist[:100, 200:].mean()

                true_and_true = a_b_dist[:100, :100].mean()

                ratio = (true_a_from_a + true_b_from_b) / (true_a_from_b + true_b_from_a)# cis / cross

                cross_a = torch.cat([a_dist[:100, torch.cat([torch.arange(100), torch.arange(200, 300)])],
                                     a_dist[200:, torch.cat([torch.arange(100), torch.arange(200, 300)])]])
                
                cross_b = torch.cat([b_dist[:100, torch.cat([torch.arange(100), torch.arange(200, 300)])],
                                    b_dist[200:, torch.cat([torch.arange(100), torch.arange(200, 300)])]])

                cross_a_b = torch.cat([a_b_dist[:100, torch.cat([torch.arange(100), torch.arange(200, 300)])],
                                       a_b_dist[200:, torch.cat([torch.arange(100), torch.arange(200, 300)])]])
                
                # print("Cross A: ", cross_a.shape)
                # print("Cross B: ", cross_b.shape)
                # print("Cross AB: ", cross_a_b.shape)

                silhouette_model_a = silhouette(same= (a_dist[100:, 100:].sum(dim=1) / 199), nearest= (a_b_dist[100:, 100:].sum(dim=1) / 200))
                silhouette_model_b = silhouette(same= (b_dist[100:, 100:].sum(dim=1) / 199), nearest= (a_b_dist[100:, 100:].sum(dim=0).T / 200))
                silhouette_true_a_from_a = silhouette(same= (a_dist[:200, :200].sum(dim=1) / 199), nearest= (a_b_dist[:200, :200].sum(dim=1) / 200))
                silhouette_true_b_from_b = silhouette(same= (b_dist[:200, :200].sum(dim=1) / 199), nearest= (a_b_dist[:200, :200].sum(dim=0).T / 200))
                silhouette_true_a_from_b = silhouette(same= (cross_a.sum(dim=1) / 199), nearest= (cross_a_b.sum(dim=1) / 200))
                silhouette_true_b_from_a = silhouette(same= (cross_b.sum(dim=1) / 199), nearest= (cross_a_b.sum(dim=0).T / 200))

                # print("SA: ", silhouette_model_a.shape)
                # print("SB: ", silhouette_model_b.shape)
                # print("STA: ", silhouette_true_a_from_a.shape)
                # print("STB: ", silhouette_true_b_from_b.shape)
                # print("STAB: ", silhouette_true_a_from_b.shape)
                # print("STBA: ", silhouette_true_b_from_a.shape)

                silhouette_model = torch.mean(torch.cat([silhouette_model_a, silhouette_model_b]))
                silhouette_true_cis = torch.mean(torch.cat([silhouette_true_a_from_a, silhouette_true_b_from_b]))
                silhouette_true_cross = torch.mean(torch.cat([silhouette_true_a_from_b, silhouette_true_b_from_a]))

                # print("Model Scikit: ", silhouette_score(torch.cat([a_out[100:, :], b_out[100:, :]]).cpu().numpy(), labels= labels))
                # print("Model Me: ", silhouette_model.item())

                # print("True Cis Scikit: ", silhouette_score(torch.cat([a_out[:200, :], b_out[:200, :]]).cpu().numpy(), labels= labels))
                # print("True Cis Me: ", silhouette_true_cis.item())

                # print("True Cross Scikit: ", silhouette_score(torch.cat(
                #                                 [a_out[torch.cat([torch.arange(100), torch.arange(200, 300)]), :],
                #                                 b_out[torch.cat([torch.arange(100), torch.arange(200, 300)]), :]]).cpu().numpy(), labels= labels))
                # print("True Cross Me: ", silhouette_true_cross.item())

                # print("Model Scikit: ", torch.tensor(silhouette_samples(torch.cat([a_out[100:, :], b_out[100:, :]]).cpu().numpy(), labels= labels)))
                # print("Model Me: ", torch.cat([silhouette_model_a, silhouette_model_b]))

                # print("True Cis Scikit: ", torch.tensor(silhouette_samples(torch.cat([a_out[:200, :], b_out[:200, :]]).cpu().numpy(), labels= labels)))
                # print("True Cis Me: ", torch.cat([silhouette_true_a_from_a, silhouette_true_b_from_b]))

                # print("True Cross Scikit: ", torch.tensor(silhouette_samples(torch.cat(
                #                                 [a_out[torch.cat([torch.arange(100), torch.arange(200, 300)]), :],
                #                                 b_out[torch.cat([torch.arange(100), torch.arange(200, 300)]), :]]).cpu().numpy(), labels= labels)))
                # print("True Cross Me: ", torch.cat([silhouette_true_a_from_b, silhouette_true_b_from_a]))

                # return

                metric_correlations[RK.CONTEXT_A] = [context_row[RK.CONTEXT_A]]
                metric_correlations[RK.CONTEXT_B] = [context_row[RK.CONTEXT_B]]

                # metric_correlations[RK.A_TO_A] = [round(a_to_a.item(), 3)]
                # metric_correlations[RK.A_TO_B] = [round(a_to_b.item(), 3)]
                # metric_correlations[RK.TO_A] = [round(to_a.item(), 3)]
                metric_correlations[RK.TRUE_A] = [round(true_a.item(), 3)]
                metric_correlations[RK.TRUE_A_AND_A_FROM_A] = [round(true_a_from_a.item(), 3)]
                metric_correlations[RK.TRUE_A_AND_A_FROM_B] = [round(true_a_from_b.item(), 3)]
                # metric_correlations[RK.RATIO_A] = [round(ratio_a.item(), 3)]
                # metric_correlations[RK.B_TO_B] = [round(b_to_b.item(), 3)]
                # metric_correlations[RK.B_TO_A] = [round(b_to_a.item(), 3)]
                # metric_correlations[RK.TO_B] = [round(to_b.item(), 3)]
                metric_correlations[RK.TRUE_B] = [round(true_b.item(), 3)]
                metric_correlations[RK.TRUE_B_AND_B_FROM_B] = [round(true_b_from_b.item(), 3)]
                metric_correlations[RK.TRUE_B_AND_B_FROM_A] = [round(true_b_from_a.item(), 3)]
                # metric_correlations[RK.RATIO_B] = [round(ratio_b.item(), 3)]
                metric_correlations[RK.TRUE_AND_TRUE] = [round(true_and_true.item(), 3)]
                metric_correlations[RK.RATIO] = [round(ratio.item(), 3)]
                # metric_correlations[RK.TRUE_RATIO] = [round(true_ratio.item(), 3)]
                metric_correlations[RK.SILHOUETTE_MODEL_A] = [round(silhouette_model_a.mean().item(), 3)]
                metric_correlations[RK.SILHOUETTE_MODEL_B] = [round(silhouette_model_b.mean().item(), 3)]
                metric_correlations[RK.SILHOUETTE_MODEL] = [round(silhouette_model.item(), 3)]
                metric_correlations[RK.SILHOUETTE_TRUE_A_FROM_A] = [round(silhouette_true_a_from_a.mean().item(), 3)]
                metric_correlations[RK.SILHOUETTE_TRUE_B_FROM_B] = [round(silhouette_true_b_from_b.mean().item(), 3)]
                metric_correlations[RK.SILHOUETTE_TRUE_CIS] = [round(silhouette_true_cis.item(), 3)]
                metric_correlations[RK.SILHOUETTE_TRUE_A_FROM_B] = [round(silhouette_true_a_from_b.mean().item(), 3)]
                metric_correlations[RK.SILHOUETTE_TRUE_B_FROM_A] = [round(silhouette_true_b_from_a.mean().item(), 3)]
                metric_correlations[RK.SILHOUETTE_TRUE_CROSS] = [round(silhouette_true_cross.item(), 3)]

                metric_correlations[RK.DIFFERENCES] = [context_row[RK.DIFFERENCES]]
                metric_correlations[RK.A_DIFFERENCES] = [out[RK.A_DIFFERENCES]]
                metric_correlations[RK.B_DIFFERENCES] = [out[RK.B_DIFFERENCES]]
                
                correlations[metric_name] = pd.concat([correlations[metric_name], metric_correlations], ignore_index=True)

        output_files = save_correlations(correlations, save_dir, n)

        for metric, path in output_files.items():
            files[metric].append(path)

    return files

def save_correlations(correlations_dict: dict[str, pd.DataFrame], save_dir: str, n: int):

    paths = {}

    for metric, correlations in correlations_dict.items():
        correlations = correlations.sort_values(RK.CONTEXT_A)

        pickle_path = os.path.join(save_dir, f"{n}_permutations_{metric}.pkl")
        correlations.to_pickle(pickle_path)
        

        correlations[[RK.DIFFERENCES, RK.A_DIFFERENCES, RK.B_DIFFERENCES]] = correlations[[RK.DIFFERENCES, RK.A_DIFFERENCES, RK.B_DIFFERENCES]].map(lambda x: " ".join(x) if len(x) > 1 else x[0])
        
        csv_path = os.path.join(save_dir, f"{n}_permutations_{metric}.csv")
        correlations.to_csv(csv_path, index=False)

        paths[metric] = pickle_path

    return paths

def plot_correlations(files: dict[str, list[str]], references_dir: str, save_dir: str):
    # context_references = pd.read_csv(os.path.join(references_dir, "context_references.csv"))
    image_files = []
    for metric, metric_files in files.items():
        for n, file in enumerate(metric_files, start=1):
            # grouped = defaultdict(lambda: defaultdict(list))
            df = pd.read_pickle(file)
            df[[RK.A_DIFFERENCES, RK.B_DIFFERENCES]] = df[[RK.A_DIFFERENCES, RK.B_DIFFERENCES]].map(lambda x: " ".join(x) if len(x) > 1 else x[0])
            df[RK.DIFFERENCES] = df[RK.DIFFERENCES].map(lambda x: "\n".join(x) if len(x) > 1 else x[0])
            # for i in range(len(df)):
            #     row = df.loc[i]
            #     modification = "\n".join(row[RK.DIFFERENCES])
            #     # context_a = row[RK.CONTEXT_A]
            #     # context_b = row[RK.CONTEXT_B]
            #     # vals_a = []
            #     # vals_b = []
            #     # for mod in row[RK.DIFFERENCES]:
            #     #     vals_a.append(context_references.loc[context_a-1][mod])
            #     #     vals_b.append(context_references.loc[context_b-1][mod])
            #     # val_a = " ".join(vals_a)
            #     # val_b = " ".join(vals_b)
            #     val_a = " ".join(row[RK.A_DIFFERENCES])
            #     val_b = " ".join(row[RK.B_DIFFERENCES])
            #     grouped[modification][val_a].append(row[RK.RATIO])
            #     grouped[modification][val_b].append(row[RK.RATIO])
            # means = []
            # for mod, data in grouped.items():
            #     for val, cors in data.items():
            #         n_samples = len(cors)
            #         means.append({"Modification Categories": mod, "Unique Value": val, f"Average {metric} Ratio": round(sum(cors) / n_samples, 3)})
            a_vals = df[[RK.DIFFERENCES, RK.A_DIFFERENCES, RK.RATIO, RK.SILHOUETTE_MODEL]].copy().rename(
                columns={RK.DIFFERENCES: "Modification Categories", RK.A_DIFFERENCES: "Unique Values"}
            )
            b_vals = df[[RK.DIFFERENCES, RK.B_DIFFERENCES, RK.RATIO, RK.SILHOUETTE_MODEL]].copy().rename(
                columns={RK.DIFFERENCES: "Modification Categories", RK.B_DIFFERENCES: "Unique Values"}
            )
            merged = pd.concat([a_vals, b_vals], ignore_index=True)
            plot_df = merged.groupby("Unique Values", as_index=False).agg({
                "Modification Categories": "first",
                RK.RATIO: "mean",
                RK.SILHOUETTE_MODEL: "mean",
            })
        
            plot_df[[RK.RATIO, RK.SILHOUETTE_MODEL]] = plot_df[[RK.RATIO, RK.SILHOUETTE_MODEL]].round(3)

            plot_df.rename(columns={RK.RATIO: f"Average {metric} Distance Ratio", RK.SILHOUETTE_MODEL: f"Average {metric} Silhouette Score"}, inplace=True)
            
            width = max(10, min(2 * plot_df["Modification Categories"].nunique(), 15))  # Keep within a reasonable range
            height = max(6, min(2 * n, 10))  # Scale height based on n
            plt.figure(figsize=(width, height))
            sns.violinplot(x="Modification Categories", y=f"Average {metric} Distance Ratio", data=plot_df, inner=None, color=".8")
            ax = sns.swarmplot(x="Modification Categories", y=f"Average {metric} Distance Ratio", data=plot_df, color="k")
            plt.title(f"Average {metric} Distance Ratio By Categories For {n} Permutation(s)")
            plt.tight_layout()
            path = os.path.join(save_dir, f"{n}_permutations_{metric}_distance.png")
            plt.savefig(path)
            plt.clf()
            plt.close()
            image_files.append(path)

            plt.figure(figsize=(width, height))
            sns.violinplot(x="Modification Categories", y=f"Average {metric} Silhouette Score", data=plot_df, inner=None, color=".8")
            ax = sns.swarmplot(x="Modification Categories", y=f"Average {metric} Silhouette Score", data=plot_df, color="k")
            plt.title(f"Average {metric} Silhouette Score By Categories For {n} Permutation(s)")
            plt.tight_layout()
            path = os.path.join(save_dir, f"{n}_permutations_{metric}_silhouette.png")
            plt.savefig(path)
            plt.clf()
            plt.close()
            image_files.append(path)

            plot_df.rename(columns={f"Average {metric} Distance Ratio": RK.RATIO, f"Average {metric} Silhouette Score": RK.SILHOUETTE_MODEL}, inplace=True)
            plot_df = plot_df[[RK.RATIO, RK.SILHOUETTE_MODEL, "Modification Categories", "Unique Values"]]
            plot_df["Modification Categories"] = plot_df["Modification Categories"].map(lambda x: x.replace("\n", " ") if len(x) > 1 else x[0])

            plot_df.to_csv(os.path.join(save_dir, f"{n}_permutations_{metric}_plot_data.csv"), index=False)

    return image_files

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
    type=click.Path(),
    required=True,
    help="Directory where the correlations are saved",
)
@click.option(
    "--skip_tensorboard", is_flag=True, help="Prevent logging plots to Tensorboard"
)
@click.pass_context
def correlations(ctx: click.Context, model_dir: str, context_data_dir: str, save_dir: str, skip_tensorboard: bool = False):
    """Run using the LightningCli."""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cross_gen = CrossGenerator(model_dir)

    files = get_correlations(cross_gen, context_data_dir, save_dir)

    # files = {"MSE": [os.path.join(save_dir, f"{n}_differences_MSE.pkl") for n in range(1,8)], "Euclidean": [os.path.join(save_dir, f"{n}_differences_Euclidean.pkl") for n in range(1,8)]}
    
    images = plot_correlations(files, context_data_dir, save_dir)

    if not skip_tensorboard:
        add_images_to_tensorboard(model_dir, images)


if __name__ == "__main__":
    correlations()
