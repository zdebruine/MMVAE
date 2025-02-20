import os
import sys
import click
import torch

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cli import CMMVAECli

FILE_PATTERN = 'human_filtered_'
SAMPLE_SIZE = 100

class CrossGenerator:

    def __init__(self, root_dir: str, ckpt_path: str = None):

        config_path = os.path.join(root_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find the config.yaml file in: {config_path}")
        
        if ckpt_path is not None:
            checkpoint = ckpt_path
        else:
            checkpoint = os.path.join(root_dir, "checkpoints", "best_model.ckpt")

        sys.argv = [sys.argv[0], "--config", config_path, "--ckpt_path", checkpoint]

        cli = CMMVAECli(run=False)
        self.model = type(cli.model).load_from_checkpoint(
            checkpoint, module=cli.model.module
        )
        self.model.module.eval()

    def _get_comparable_contexts(self, reference_dataframe: pd.DataFrame):
        comparable_contexts = defaultdict(dict)
        for i in range(len(reference_dataframe)):
            row = reference_dataframe.iloc[i]
            for j in range(i + 1, len(reference_dataframe)):
                other = reference_dataframe.iloc[j]
                mask = row[RK.FILTER_CATEGORIES] == other[RK.FILTER_CATEGORIES]
                if mask.sum() == len(RK.FILTER_CATEGORIES) - 1:
                    difference = mask.index[~mask].tolist()[0]
                    comparable_contexts[row['group_id']][other['group_id']] = {"mod_type": difference, "mod_value": other[difference]}
        return comparable_contexts

    def get_contexts(self, path: str):
        df = pd.read_csv(path)
        return self._get_comparable_contexts(df)

    def _convert_to_tensor(self, data: sp.csr_matrix, return_dense: bool = True):
        tensor = torch.sparse_csr_tensor(
            crow_indices=data.indptr,
            col_indices=data.indices,
            values=data.data,
            size=data.shape,
            dtype=torch.float32,
        )
        if return_dense:
            tensor = tensor.to_dense()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def _get_data(self, data_dir: str, contex_id: int):
        data = sp.load_npz(
            os.path.join(
                data_dir, f'{FILE_PATTERN}{contex_id}.npz'
            )
        )
        metadata = pd.read_pickle(
            os.path.join(
                data_dir, f'{FILE_PATTERN}{contex_id}.pkl'
            )
        )
        if data.shape[0] > SAMPLE_SIZE:
            sample = np.random.choice(data.shape[0], SAMPLE_SIZE, replace=False)
            data = data[sample, :]
            metadata = metadata.iloc[sample]
        data = self._convert_to_tensor(data)
        return data, metadata

    @torch.no_grad()
    def _get_xhat(
        self,
        z: torch.Tensor,
        metadata: pd.DataFrame,
    ):
        xhat = self.model.module.vae.after_reparameterize(z, metadata, species=RK.HUMAN)
        xhat = self.model.module.vae.decode(xhat)
        xhat = self.model.module.experts[RK.HUMAN].decode(xhat)
        
        return xhat

    @torch.no_grad()
    def _get_z(
        self,
        x: torch.Tensor,
    ):
        x = self.model.module.experts[RK.HUMAN].encode(x)
        _, z, _ = self.model.module.vae.encode(x)
        
        return z

    @torch.no_grad()
    def get_cis_outputs(
        self,
        x: torch.Tensor,
        metadata: pd.DataFrame,
        return_z: bool = True,
        transpose: bool = False,
    ):

        z = self._get_z(x)
        xhat = self._get_xhat(z, metadata)
        print(xhat.shape, transpose)
        if transpose:
            xhat.transpose_(0, 1)
        print(xhat.shape, transpose)

        if return_z:
            return xhat, z
        else:
            return xhat

    @torch.no_grad()
    def get_cross_outputs(
        self,
        z: torch.Tensor,
        source_metadata: pd.DataFrame,
        target_metadata: pd.DataFrame,
        mod_tags: list[str],
        transpose: bool = False,
    ):
        modified_metadata = source_metadata.copy(deep=True)

        if isinstance(mod_tags, str):
            mod_tags = [mod_tags]

        for mod_tag in mod_tags:
            modified_metadata[mod_tag] = target_metadata[mod_tag].iloc[0]

        xhat = self._get_xhat(z, modified_metadata)

        if transpose:
            xhat.transpose_(0, 1)

        return xhat

    def cross_generate(
        self,
        data_dir: str,
        primary_context: int,
        contexts: dict[int, dict[str, str]],
        transpose: bool = False,
    ):
        generations = {}
        print("Cross-gen transpose", transpose)
        primary_x, primary_metadata = self._get_data(data_dir, primary_context)
        primary_cis_xhat, primary_z = self.get_cis_outputs(primary_x, primary_metadata, transpose= transpose)
        
        primary_x = primary_x.cpu()
        generations[f"{primary_context}_to_{primary_context}"] = primary_cis_xhat
        generations["metadata"] = primary_metadata

        for secondary_context, modifications in contexts.items():
            secondary_x, secondary_metadata = self._get_data(data_dir, secondary_context)
            secondary_cis_xhat, secondary_z = self.get_cis_outputs(secondary_x, secondary_metadata, transpose= transpose)
            
            secondary_x = secondary_x.cpu()
            generations[f"{secondary_context}_to_{secondary_context}"] = secondary_cis_xhat

            primary_cross_xhat = self.get_cross_outputs(
                z= primary_z,
                source_metadata= primary_metadata,
                target_metadata= secondary_metadata,
                mod_tags= modifications["mod_type"],
                transpose = transpose,
            )
            generations[f"{primary_context}_to_{secondary_context}"] = primary_cross_xhat

            secondary_cross_xhat = self.get_cross_outputs(
                z= secondary_z,
                source_metadata= secondary_metadata,
                target_metadata= primary_metadata,
                mod_tags= modifications["mod_type"],
                transpose = transpose,
            )
            generations[f"{secondary_context}_to_{primary_context}"] = secondary_cross_xhat
        
        return generations

# @click.command(
#     context_settings=dict(
#         ignore_unknown_options=True,
#         allow_extra_args=True,
#     )
# )
# @click.option(
#     "--context_data_dir",
#     type=click.Path(exists=True),
#     required=True,
#     help="Directory where the filtered context data is stored",
# )
# @click.option(
#     "--context_references",
#     type=click.Path(exists=True),
#     required=True,
#     help="Path to the context references file",
# )
# @click.option(
#     "--save_dir",
#     type=click.Path(exists=True),
#     required=True,
#     help="Directory where the correlations are saved",
# )
# @click.pass_context
# def cross_generation(ctx: click.Context, context_data_dir: str, context_references: str, save_dir: str):

#     """Run using the LightningCli."""
#     if ctx.args:
#         # Ensure `args` is passed as the command-line arguments
#         sys.argv = [sys.argv[0]] + ctx.args
    
#     print(sys.argv)
#     cli = CMMVAECli(run=False)
#     model = type(cli.model).load_from_checkpoint(
#         cli.config["ckpt_path"], module=cli.model.module
#     )

#     # contexts = get_contexts(context_references)
#     # correlations = cross_generate(model, contexts, context_data_dir)
#     # save_correlations(correlations, save_dir)


# if __name__ == "__main__":
#     cross_generation()
