import torch
import pytest

from cmmvae.models.base_model import tag_log_dict


def tensors_equal(tensor1, tensor2):
    if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
        return torch.equal(tensor1, tensor2)
    return tensor1 == tensor2


def dicts_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    return all(tensors_equal(dict1[key], dict2[key]) for key in dict1)


def test_tag_log_dict_no_tags():
    log_dict = {"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.9)}
    result = tag_log_dict(log_dict)
    expected = {"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.9)}
    assert dicts_equal(result, expected)


def test_tag_log_dict_with_tags_first():
    log_dict = {"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.9)}
    tags = ["modelA", "experiment1"]
    result = tag_log_dict(log_dict, tags=tags, sep="_", key_pos="first")
    expected = {
        "loss_modelA_experiment1": torch.tensor(1.0),
        "accuracy_modelA_experiment1": torch.tensor(0.9),
    }
    assert dicts_equal(result, expected)


def test_tag_log_dict_with_tags_last():
    log_dict = {"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.9)}
    tags = ["modelA", "experiment1"]
    result = tag_log_dict(log_dict, tags=tags, sep="_", key_pos="last")
    expected = {
        "modelA_experiment1_loss": torch.tensor(1.0),
        "modelA_experiment1_accuracy": torch.tensor(0.9),
    }
    assert dicts_equal(result, expected)


def test_tag_log_dict_empty_dict():
    log_dict = {}
    result = tag_log_dict(log_dict)
    expected = {}
    assert dicts_equal(result, expected)


def test_tag_log_dict_invalid_key_pos():
    log_dict = {"loss": torch.tensor(1.0)}
    with pytest.raises(ValueError):
        tag_log_dict(log_dict, key_pos="invalid")
