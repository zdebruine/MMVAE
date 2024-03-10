import pytest
import torch
from unittest.mock import patch, MagicMock

from mmvae.trainers.HumanVAE import HumanVAETrainer, HumanVAEConfig

@pytest.fixture
def mock_device():
    return torch.device('cpu')

@pytest.fixture
def mock_hparams():
    # Prepare a dictionary with all necessary hyperparameters
    hparams_dict = {
        'data_file_path': 'path/to/data_file',
        'metadata_file_path': 'path/to/metadata_file',
        'train_dataset_ratio': 0.8,
        'batch_size': 64,
        'expert.encoder.optimizer.lr': 1e-3,
        'expert.decoder.optimizer.lr': 1e-3,
        'shr_vae.optimizer.lr': 1e-3,
        'kl_cyclic.warm_start': 10,
        'kl_cyclic.cycle_length': 100.0,
        'kl_cyclic.min_beta': 0.0,
        'kl_cyclic.max_beta': 1.0,
        'snapshot.path': None,
        'snapshot.save_every': None,
        'tensorboard.directory': None,
        'tensorboard.run_name': None,
    }
    return HumanVAEConfig(hparams_dict)

@pytest.fixture
def trainer(mock_device, mock_hparams):
    # Patching external dependencies during initialization
    with patch('mmvae.trainers.HumanVAE.HumanVAETrainer.configure_model'), \
         patch('mmvae.trainers.HumanVAE.HumanVAETrainer.configure_dataloader'), \
         patch('mmvae.trainers.HumanVAE.HumanVAETrainer.configure_optimizers'), \
         patch('mmvae.trainers.HumanVAE.HumanVAETrainer.configure_schedulers'):
        trainer_instance = HumanVAETrainer(mock_device, mock_hparams)
    return trainer_instance

# Test the initialization of the trainer
def test_initialization(trainer):
    assert trainer is not None

# Test the configuration of data loaders
@patch('mmvae.data.configure_singlechunk_dataloaders', return_value=(MagicMock(), MagicMock()))
def test_configure_dataloader(mock_loader, trainer):
    trainer.configure_dataloader()
    mock_loader.assert_called_once_with(
        data_file_path=trainer.hparams['data_file_path'],
        metadata_file_path=trainer.hparams['metadata_file_path'],
        train_ratio=trainer.hparams['train_dataset_ratio'],
        batch_size=trainer.hparams['batch_size'],
        device=trainer.device
    )
    assert trainer.train_loader is not None
    assert trainer.test_loader is not None

# Mocking a model and its components for reconstruction and training tests
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = MagicMock()
        self.shared_vae = MagicMock()
        self.expert.encoder = MagicMock()
        self.expert.decoder = MagicMock()
        self.parameters = MagicMock(return_value=[torch.randn(1, requires_grad=True)])

    def forward(self, x):
        return torch.randn_like(x), torch.randn(x.size(0), 10), torch.randn(x.size(0), 10)

def test_prevent_model_override(trainer):
    with pytest.raises(RuntimeError) as excinfo:
        trainer.model = MockModel()
    assert "cannot be set after initialization" in str(excinfo.value)

def test_prevent_optimizers_override(trainer):
    with pytest.raises(RuntimeError) as excinfo:
        trainer.optimizers = MockModel()
    assert "cannot be set after initialization" in str(excinfo.value)

def test_prevent_schedulars_override(trainer):
    with pytest.raises(RuntimeError) as excinfo:
        trainer.schedulars = MockModel()
    assert "cannot be set after initialization" in str(excinfo.value)
