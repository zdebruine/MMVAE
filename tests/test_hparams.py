import pytest
from mmvae.trainers.hparams import HPConfig

class ExampleHPConfig(HPConfig):
    required_hparams = {
        'example.int': int,
        'example.str': str,
        'nested.example.float': float
    }

@pytest.fixture
def example_config():
    return {
        'example': {
            'int': 10,
            'str': 'test',
        },
        'nested': {
            'example': {
                'float': 2.0
            }
        }
    }

def test_initialization_and_flattening(example_config):
    hp = ExampleHPConfig(example_config)
    assert hp['example.int'] == 10
    assert hp['example.str'] == 'test'
    assert hp['nested.example.float'] == 2.0

def test_required_hparams_missing():
    config_missing_required = {
        'example': {
            'int': 10
        }
    }
    with pytest.raises(ValueError) as excinfo:
        ExampleHPConfig(config_missing_required)
    assert "not in supplied hparams" in str(excinfo.value)

def test_hparam_type_validation():
    config_invalid_type = {
        'example': {
            'int': 'not an int',
            'str': 'test',
        },
        'nested': {
            'example': {
                'float': 2.0
            }
        }
    }
    with pytest.raises(TypeError) as excinfo:
        ExampleHPConfig(config_invalid_type)
    assert "expected type" in str(excinfo.value)

def test_protected_attribute_setting(example_config):
    hp = ExampleHPConfig(example_config)
    with pytest.raises(RuntimeError) as excinfo:
        hp.config = {}
    assert "Attribute config cannot be set after runtime" in str(excinfo.value)

def test_attribute_setting_post_initialization(example_config):
    hp = ExampleHPConfig(example_config)
    # Test setting a new attribute
    hp.new_attribute = "new value"
    assert hp.new_attribute == "new value"
    # Test modifying an existing attribute not protected
    hp['example.int'] = 20
    assert hp['example.int'] == 20
