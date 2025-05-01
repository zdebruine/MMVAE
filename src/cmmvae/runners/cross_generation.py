import os
import sys
import itertools
import torch

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.runners.cli import CMMVAECli

FILE_PATTERN = 'human_filtered_'
SAMPLE_SIZE = 1000

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

    def get_data(self, data_dir: str, filename: str, sample: bool = True):
        data = sp.load_npz(
            os.path.join(
                data_dir, f'{filename}_counts.npz'
            )
        )
        metadata = pd.read_pickle(
            os.path.join(
                data_dir, f'{filename}_metadata.pkl'
            )
        )
        if sample and data.shape[0] > SAMPLE_SIZE:
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
    ):

        z = self._get_z(x)
        xhat = self._get_xhat(z, metadata)

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
        mod_tags: tuple[str],
    ):
        modified_metadata = source_metadata.copy(deep=True)

        if isinstance(mod_tags, str):
            mod_tags = [mod_tags]

        for mod_tag in mod_tags:
            modified_metadata[mod_tag] = target_metadata[mod_tag].iloc[0]

        xhat = self._get_xhat(z, modified_metadata)

        return xhat

    def cross_generate(
        self,
        data_dir: str,
        contexts: pd.DataFrame,
        return_real: bool = False,
    ):
        generations = {}

        context_a_x, context_a_metadata = self.get_data(data_dir, contexts[RK.CONTEXT_A])
        context_a_to_a_xhat, context_a_z = self.get_cis_outputs(context_a_x, context_a_metadata)
        
        if return_real:
            generations[RK.TRUE_A] = context_a_x
        else:
            context_a_x = context_a_x.cpu()
        
        generations[RK.A_TO_A] = context_a_to_a_xhat

        context_b_x, context_b_metadata = self.get_data(data_dir, contexts[RK.CONTEXT_B])
        context_b_to_b_xhat, context_b_z = self.get_cis_outputs(context_b_x, context_b_metadata)
        
        if return_real:
            generations[RK.TRUE_B] = context_b_x
        else:
            context_b_x = context_b_x.cpu()

        generations[RK.B_TO_B] = context_b_to_b_xhat

        context_a_to_b_xhat = self.get_cross_outputs(
            z= context_a_z,
            source_metadata= context_a_metadata,
            target_metadata= context_b_metadata,
            mod_tags= contexts[RK.DIFFERENCES],
        )
        generations[RK.A_TO_B] = context_a_to_b_xhat

        context_b_to_a_xhat = self.get_cross_outputs(
            z= context_b_z,
            source_metadata= context_b_metadata,
            target_metadata= context_a_metadata,
            mod_tags= contexts[RK.DIFFERENCES],
        )
        generations[RK.B_TO_A] = context_b_to_a_xhat

        generations[RK.A_DIFFERENCES] = context_a_metadata[contexts[RK.DIFFERENCES]].values[0]
        generations[RK.B_DIFFERENCES] = context_b_metadata[contexts[RK.DIFFERENCES]].values[0]

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
