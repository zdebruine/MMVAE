import pytest
from pytest import param, raises
from contextlib import nullcontext as does_not_raise



from d_mmvae.Models import Encoder, Decoder, VAE



class TestVAE:
    @pytest.mark.parametrize(
        argnames="enc_type, dec_type, expectation",
        argvalues=[
            ( Encoder, Decoder, does_not_raise() ),
            ( Encoder, Encoder, raises( TypeError ) ),
            ( Decoder, Decoder, raises( TypeError ) )
        ]
    )
    def test_encoder_type( self, enc_type, dec_type, expectation ):
        with expectation:
            _ = VAE( enc_type( 1, 1, 1 ), dec_type( 1, 1, 1 ) )