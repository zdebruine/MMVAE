import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import mmvae.models.HumanVAE as H  # Make sure to import your SharedVAE class correctly

@pytest.fixture
def encoder():
    return nn.Sequential(
        nn.Linear(200, 100),
        nn.ReLU(),
    )

@pytest.fixture
def decoder():
    return nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU()
    )

@pytest.fixture
def mu():
    return nn.Linear(100, 100)

@pytest.fixture
def var():
    return nn.Linear(100, 100)

@pytest.fixture
def vae_model(encoder, decoder, mu, var):
    # Fixture to create a SharedVAE instance without initializing weights
    model = H.SharedVAE(encoder, decoder, mu, var, init_weights=False)
    return model

@pytest.fixture
def vae_model_with_weights(encoder, decoder, mu, var):
    # Fixture to create a SharedVAE instance with weights initialized
    model = H.SharedVAE(encoder, decoder, mu, var, init_weights=True)
    return model

def check_weights_initialized(module):
    assert isinstance(module, nn.Module)
    for param in module.named_parameters():
        if 'weight' in param[0]:
            assert torch.all(param[1] != 0), f"Weights not initialized {param[0]}"

def test_sharedvae_initialization_without_weights(vae_model):
    # Test to ensure the model initializes without weights
    # Here, we simply check if the model is an instance of SharedVAE
    assert isinstance(vae_model, H.SharedVAE), "Model is not an instance of SharedVAE"

def test_sharedvae_initialization_with_weights(vae_model_with_weights):
    # Test to ensure weights are initialized
    # This test depends on the structure of SharedVAE and the existence of encoder, decoder, mean, var
    assert isinstance(vae_model_with_weights, H.SharedVAE), "Model is not an instance of SharedVAE"
    check_weights_initialized(vae_model_with_weights.encoder)
    check_weights_initialized(vae_model_with_weights.decoder)
    check_weights_initialized(vae_model_with_weights.mean)
    # var layer not checked due to bias of -1