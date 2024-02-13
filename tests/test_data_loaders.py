import pytest
import torch
from unittest.mock import MagicMock
from mmvae.data.loaders import MultiModalLoader

def test_initialization_with_single_modal():
    mock_modal = MagicMock()
    loader = MultiModalLoader(mock_modal)
    assert len(loader.modals) == 1
    assert loader.exhaust_all == True

def test_initialization_with_multiple_modals():
    mock_modal1 = MagicMock()
    mock_modal2 = MagicMock()
    loader = MultiModalLoader(mock_modal1, mock_modal2, exhaust_all=False)
    assert len(loader.modals) == 2
    assert loader.exhaust_all == False

def test_len_before_iteration():
    mock_modal = MagicMock()
    loader = MultiModalLoader(mock_modal)

    with pytest.raises(RuntimeError) as excinfo:
        len(loader)
    assert "cannot be determined until one entire forward pass" in str(excinfo.value)

def test_iteration_with_single_modal():
    mock_modal = MagicMock()
    mock_modal.__iter__.return_value = iter([(torch.tensor([1, 2, 3]), 'data', None)])
    loader = MultiModalLoader(mock_modal)

    for data, _, _ in loader:
        assert torch.equal(data, torch.tensor([1, 2, 3]))
    assert len(loader) == 1

def test_iteration_with_multiple_modals():
    mock_modal1 = MagicMock()
    mock_modal1.__iter__.return_value = iter([(torch.tensor([1, 2, 3]), 'data1', None)])
    
    mock_modal2 = MagicMock()
    mock_modal2.__iter__.return_value = iter([(torch.tensor([4, 5, 6]), 'data2', None)])

    loader = MultiModalLoader(mock_modal1, mock_modal2)
    
    data_list = [data for data, _, _ in loader]
    assert len(data_list) >= 1  # since it's stochastic, we can't assert the exact length
    assert len(loader) == len(data_list)

def test_exhaust_all_false():
    mock_modal1 = MagicMock()
    mock_modal1.__iter__.return_value = iter([(torch.tensor([1, 2, 3]), 'data1', None)])
    
    mock_modal2 = MagicMock()
    mock_modal2.__iter__.return_value = iter([]) 

    loader = MultiModalLoader(mock_modal1, mock_modal2, exhaust_all=False)
    
    l = iter(loader)
    with pytest.raises(StopIteration):
        while True:
            next(l)
        
def test_initialization_with_no_modals():
    with pytest.raises(ValueError) as excinfo:
        loader = MultiModalLoader()

def test_unexpected_exception():
    mock_modal = MagicMock()
    mock_modal.__iter__.side_effect = Exception("Unexpected error")
    loader = MultiModalLoader(mock_modal)

    with pytest.raises(Exception) as excinfo:
        next(iter(loader))
    assert "Unexpected error" in str(excinfo.value)
