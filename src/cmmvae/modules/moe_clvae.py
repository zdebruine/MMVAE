import torch.nn as nn
from cmmvae.modules.clvae import CLVAE


class MOE_CLVAE(CLVAE):
    """
    Mixture of Experts Conditional Latent Variational Autoencoder class.

    This class extends the CLVAE class, used to create human and
    mouse specific variational autoencoders.

    Args:
        encoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the encoder's fully connected block.
        decoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the decoder's fully connected block.
        conditional_config (Optional[cmmvae.modules.base.FCBlockConfig]):
            Configuration for the conditional layers.
        conditional_paths (Dict[str, str]):
            Mapping of conditional paths for the conditional layers.
        selection_order (Optional[List[str]]):
            Order in which to apply selection of conditionals.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """

    def __init__(self, id: str, **clvae_kwargs):
        super().__init__(**clvae_kwargs)
        self.id = id


class ExpertVAEs(nn.ModuleDict):
    """
    Container to store human and mouse specific VAEs.

    Args:
        vaes (list[cmmvae.modules.base.BaseExpert]):
            List of Expert modules.

    Attributes:
        labels (dict[str, int]):
            Dictionary of expert.id's to their integer representation.
    """

    def __init__(self, vaes: list[MOE_CLVAE]):
        super().__init__({vae.id: vae for vae in vaes if isinstance(vae, MOE_CLVAE)})
        self.labels = {key: i for i, key in enumerate(self.keys())}
