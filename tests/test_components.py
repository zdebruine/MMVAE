
# test_neural_network_components.py

import pytest
import torch
import torch.nn as nn
import os
import pandas as pd
from cmmvae.modules.base.components import (
    is_iterable,
    BaseFCBlock,
    FCBlockConfig,
    ConditionalLayer,
    ConditionalLayers,
    Encoder,
    BaseExpert,
    Expert,
    Experts,
)

# 1. Utility Function: is_iterable
def test_is_iterable_with_list():
    assert is_iterable([1, 2, 3]) == True

def test_is_iterable_with_string():
    assert is_iterable("string") == True

def test_is_iterable_with_integer():
    assert is_iterable(123) == False

def test_is_iterable_with_dict():
    assert is_iterable({"key": "value"}) == True

def test_is_iterable_with_none():
    assert is_iterable(None) == False


# 2. Class: BaseFCBlock
def test_base_fc_block_initialization():
    block = BaseFCBlock(layers=[10, 20, 30], dropout_rate=0.5)
    assert block.n_layers == 2
    assert block.input_dim == 10
    assert block.output_dim == 30

def test_base_fc_block_forward_pass():
    block = BaseFCBlock(layers=[10, 20, 30], dropout_rate=0.5)
    input_tensor = torch.randn(5, 10)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (5, 30)

def test_base_fc_block_can_bypass():
    block = BaseFCBlock(layers=[10, 20, 30], return_hidden=False)
    assert block.can_bypass == True

def test_base_fc_block_return_hidden():
    block = BaseFCBlock(layers=[10, 20, 30], return_hidden=True)
    assert block.can_bypass == False


# 3. Class: FCBlockConfig
def test_fc_block_config_initialization():
    config = FCBlockConfig(layers=[10, 20, 30], dropout_rate=[0.5, 0.3], use_batch_norm=True)
    assert config.n_layers == 2
    assert config.layers == [10, 20, 30]
    assert config.dropout_rate == [0.5, 0.3]

def test_fc_block_config_activation_function():
    config = FCBlockConfig(layers=[10, 20], activation_fn=nn.ReLU)
    assert all(issubclass(act, nn.ReLU) for act in config.activation_fn)

def test_fc_block_config_invalid_layers():
    with pytest.raises(AssertionError):
        FCBlockConfig(layers=[-10, 20])  # Invalid layer sizes

def test_fc_block_config_mismatched_lengths():
    with pytest.raises(ValueError):
        FCBlockConfig(layers=[10, 20, 30], dropout_rate=[0.5])  # Mismatched lengths


# 4. Classes: ConditionalLayer and ConditionalLayers
def test_conditional_layer_initialization():
    config = FCBlockConfig(layers=[10])
    layer = ConditionalLayer(batch_key='assay', conditions_path=f'{os.getcwd()}/src/cmmvae/data/conditional_layers/unique_assays.csv', fc_block_config=config)
    assert layer.batch_key == 'assay'
    assert len(layer.conditions) == len(layer.unique_conditions)

def test_conditional_layer_forward_pass():
    config = FCBlockConfig(layers=[10])
    layer = ConditionalLayer(batch_key='assay', conditions_path=f'{os.getcwd()}/src/cmmvae/data/conditional_layers/unique_assays.csv', fc_block_config=config)
    x = torch.randn(5, 10)
    metadata = pd.DataFrame({'assay': ['10x 5\' v1', '10x 3\' v3', 'microwell-seq', 'microwell-seq', '10x 5\' transcription profiling']})
    output = layer(x, metadata)
    assert output.shape == x.shape

def test_conditional_layers_initialization():
    config = FCBlockConfig(layers=[10])
    conditional_paths = {'assay': f'{os.getcwd()}/src/cmmvae/data/conditional_layers/unique_assays.csv', 'sex': f'{os.getcwd()}/src/cmmvae/data/conditional_layers/unique_sex.csv'}
    layers = ConditionalLayers(conditional_paths, fc_block_config=config)
    assert len(layers.layers) == 2

def test_conditional_layers_forward_pass():
    config = FCBlockConfig(layers=[10])
    conditional_paths = {'assay': f'{os.getcwd()}/src/cmmvae/data/conditional_layers/unique_assays.csv', 'sex': f'{os.getcwd()}/src/cmmvae/data/conditional_layers/unique_sex.csv'}
    layers = ConditionalLayers(conditional_paths, fc_block_config=config)
    x = torch.randn(5, 10)
    metadata = pd.DataFrame({
        'assay': ['10x 5\' v1', '10x 3\' v3', 'microwell-seq', 'microwell-seq', '10x 5\' transcription profiling'],
        'sex': ['male', 'female', 'male', 'female', 'male']
    })
    output = layers(x, metadata)
    assert output.shape == x.shape


# 5. Class: Encoder
def test_encoder_initialization():
    config = FCBlockConfig(layers=[10])
    encoder = Encoder(latent_dim=5, fc_block_config=config)
    assert encoder.n_layers == 1
    assert encoder.var_eps == 1e-4

def test_encoder_forward_pass():
    config = FCBlockConfig(layers=[10])
    encoder = Encoder(latent_dim=5, fc_block_config=config)
    x = torch.randn(5, 10)
    q_m, q_v, latent, hidden_representations = encoder(x)
    assert q_m.shape == (5, 5)
    assert q_v.shape == (5, 5)
    assert latent.shape == (5, 5)
    assert isinstance(hidden_representations, list)


# 6. Class: BaseExpert and Expert
def test_base_expert_initialization():
    config = FCBlockConfig(layers=[10, 20])
    encoder = Expert('expert1', encoder_config=config, decoder_config=config)
    assert encoder.id == 'expert1'
    assert encoder.encoder is not None
    assert encoder.decoder is not None

def test_base_expert_encode_decode():
    config = FCBlockConfig(layers=[10])
    expert = Expert('expert1', encoder_config=config, decoder_config=config)
    x = torch.randn(5, 10)
    encoded = expert.encode(x)
    decoded = expert.decode(encoded)  # Pass the output from encode to decode
    assert encoded.shape == (5, 10)
    assert decoded.shape == (5, 10)
    
def test_base_expert_encode_decode_return_hidden():
    encoder_config = FCBlockConfig(layers=[10], return_hidden=[True])
    decoder_config = FCBlockConfig(layers=[10])
    expert = Expert('expert1', encoder_config=encoder_config, decoder_config=decoder_config)
    x = torch.randn(5, 10)
    encoded, hidden = expert.encode(x)
    decoded = expert.decode(encoded)  # Pass the output from encode to decode
    assert all(h.shape == (5, 10) for h in hidden)
    assert encoded.shape == (5, 10)
    assert decoded.shape == (5, 10)


# 7. Class: Experts
def test_experts_initialization():
    config = FCBlockConfig(layers=[10, 20])
    expert1 = Expert('expert1', encoder_config=config, decoder_config=config)
    expert2 = Expert('expert2', encoder_config=config, decoder_config=config)
    experts = Experts([expert1, expert2])
    assert len(experts) == 2
    assert 'expert1' in experts
    assert 'expert2' in experts

def test_experts_labels():
    config = FCBlockConfig(layers=[10, 20])
    expert1 = Expert('expert1', encoder_config=config, decoder_config=config)
    expert2 = Expert('expert2', encoder_config=config, decoder_config=config)
    experts = Experts([expert1, expert2])
    assert experts.labels['expert1'] == 0
    assert experts.labels['expert2'] == 1
