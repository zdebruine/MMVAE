import pytest
from pytest import param, raises
from contextlib import nullcontext as does_not_raise



from d_mmvae.Models import Decoder



class TestDecoder:    # Parameterization does not work in subclasses.
    @pytest.mark.parametrize(
        argnames="dim, expectation",
        argvalues=[
            (  1, does_not_raise() ),
            (  0, raises( ValueError ) ),
            ( -1, raises( ValueError ) )
        ]
    )
    class TestDecoderArgValues:
        def test_latent_dim( self, dim, expectation ):
            with expectation:
                _ = Decoder( dim, 1, 1 )
        def test_hidden_dim( self, dim, expectation ):
            with expectation:
                _ = Decoder( 1, dim, 1 )
        def test_output_dim( self, dim, expectation ):
            with expectation:
                _ = Decoder( 1, 1, dim )
    
    @pytest.mark.parametrize(
        argnames="dim, expectation",
        argvalues=[
            ( int( 1 ), does_not_raise() ),
            ( float( 1 ), raises( TypeError ) ),
            ( str( 1 ), raises( TypeError ) )
        ]
    )
    class TestDecoderArgTypes:
        def test_latent_dim( self, dim, expectation ):
            with expectation:
                _ = Decoder( dim, 1, 1 )
        def test_hidden_dim( self, dim, expectation ):
            with expectation:
                _ = Decoder( 1, dim, 1 )
        def test_output_dim( self, dim, expectation ):
            with expectation:
                _ = Decoder( 1, 1, dim )