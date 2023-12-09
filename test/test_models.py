from unittest import TestCase
from torch import Size
from torch.testing import assert_close
# from torch.testing._internal.common_utils import TestCase as TorchTestCase
# from torch.testing._internal.common_device_type import instantiate_device_type_tests


from d_mmvae.Dataset import CellxGeneDataset
from d_mmvae.Models import Decoder, Encoder, VAE


HIDDEN_LEN = 512
INPUT_LEN = 60664
LATENT_LEN = 32


class TestModels( TestCase ):
    def setUp(self) -> None:
        self.encoder = Encoder(
            input_dim=INPUT_LEN,
            hidden_dims=HIDDEN_LEN,
            latent_dim=LATENT_LEN
        )
        self.decoder = Decoder(
            latent_dim=LATENT_LEN,
            hidden_dims=HIDDEN_LEN,
            output_dim=INPUT_LEN
        )
        self.vae = VAE( self.encoder, self.decoder )
        self.sample = CellxGeneDataset( 32 )[ 0 ]

    def test_vae( self ):
        output, _, _ = self.vae( self.sample )
        assert_close( output, self.sample.to_dense() )