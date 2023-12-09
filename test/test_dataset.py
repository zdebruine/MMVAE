from unittest import TestCase
from torch import Size
from torch.testing import assert_close
# from torch.testing._internal.common_utils import TestCase as TorchTestCase
# from torch.testing._internal.common_device_type import instantiate_device_type_tests


from d_mmvae.Dataset import CellxGeneDataset


"""
Check...
    - Dataset length
    - Sample shape
    - 
"""


BATCH_SIZE = 32
N_CHUNKS = 10
N_SAMPLES = 285341
SAMPLE_LEN = 60664


class TestDataset( TestCase ):
    def setUp( self ) -> None:
        self.dataset = CellxGeneDataset(
            batch_size=BATCH_SIZE,
            buffer_size=6
        )

    def test_dataset_length( self ):
        expected_len = N_SAMPLES // BATCH_SIZE * N_CHUNKS
        dataset_len = len( self.dataset )
        self.assertEqual(
            dataset_len, expected_len,
            f"Incorrect length; {dataset_len} != {expected_len}"
        )

    def test_sample_shape( self ):
        expected_shape = Size( ( BATCH_SIZE, SAMPLE_LEN ) )
        sample_shape = self.dataset[ 0 ].shape
        self.assertEqual(
            sample_shape, expected_shape,
            f"Incorrect shape; {sample_shape} != {expected_shape}"
        )
