import pytest
from pytest import param, raises
from contextlib import nullcontext as does_not_raise



from d_mmvae.models.Models import Encoder



class TestEncoder:    # Parameterization does not work in subclasses.
    @pytest.mark.parametrize(
        argnames="dim, expectation",
        argvalues=[
            (  1, does_not_raise() ),
            (  0, raises( ValueError ) ),
            ( -1, raises( ValueError ) )
        ]
    )
    class TestEncoderArgValues:
        def test_input_dim( self, dim, expectation ):
            with expectation:
                _ = Encoder( dim, 1, 1 )
        def test_hidden_dim( self, dim, expectation ):
            with expectation:
                _ = Encoder( 1, dim, 1 )
        def test_latent_dim( self, dim, expectation ):
            with expectation:
                _ = Encoder( 1, 1, dim )
    
    @pytest.mark.parametrize(
        argnames="dim, expectation",
        argvalues=[
            ( int( 1 ), does_not_raise() ),
            ( float( 1 ), raises( TypeError ) ),
            ( str( 1 ), raises( TypeError ) )
        ]
    )
    class TestEncoderArgTypes:
        def test_input_dim( self, dim, expectation ):
            with expectation:
                _ = Encoder( dim, 1, 1 )
        def test_hidden_dim( self, dim, expectation ):
            with expectation:
                _ = Encoder( 1, dim, 1 )
        def test_latent_dim( self, dim, expectation ):
            with expectation:
                _ = Encoder( 1, 1, dim )