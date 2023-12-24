from unittest import TestCase
from torch import Size
import torch
from torch.testing import assert_close
# from torch.testing._internal.common_utils import TestCase as TorchTestCase
# from torch.testing._internal.common_device_type import instantiate_device_type_tests



from d_mmvae.Dataset import CellxGeneDataset



BATCH_SIZE = 32
N_CHUNKS = 10
N_SAMPLES = 285341
SAMPLE_LEN = 60664



class TestDataset:
    def setup_class( self ):
        self.dataset = CellxGeneDataset( batch_size=BATCH_SIZE )[:100]    # Using a slice to speed up tests for now.

    # def test_dataset_length( self ):
    #     expected_len = N_SAMPLES // BATCH_SIZE * N_CHUNKS
    #     dataset_len = len( self.dataset )
    #     assert dataset_len == expected_len

    # def test_sample_shape( self ):
    #     expected_shape = Size( ( BATCH_SIZE, SAMPLE_LEN ) )
    #     sample_shape = self.dataset[ 0 ].shape
    #     assert sample_shape == expected_shape
    
    def test_dataset_scaling_lt1( self ):
        for sample in self.dataset:
            dense_sample = sample.to_dense()
            assert dense_sample.max() <= 1
    def test_dataset_scaling_gtneg1( self ):
        for sample in self.dataset:
            dense_sample = sample.to_dense()
            assert dense_sample.min() >= -1
    def test_dataset_scaling_gt0( self ):
        for sample in self.dataset:
            dense_sample = sample.to_dense()
            assert torch.any( dense_sample > 0 ) == True
    def test_dataset_scaling_lt0( self ):
        for sample in self.dataset:
            dense_sample = sample.to_dense()
            assert torch.any( dense_sample < 0 ) == True