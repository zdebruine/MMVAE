import torch
import random
from torch import float32, nn
from d_mmvae.data import CellCensusDataLoader

DATA_SIZE = 60664
REAL_DATA = 1
FAKE_DATA = 0

human_data = CellCensusDataLoader('human_data', directory_path='/active/debruinz_project/human_data/python_data', masks=['human_chunk*'], batch_size=64, num_workers=2)
mouse_data = CellCensusDataLoader('human_data', directory_path='/active/debruinz_project/mouse_data/python_data', masks=['mouse_chunk*'], batch_size=64, num_workers=2)

human_expert_discriminator = nn.Sequential(
    nn.Linear(DATA_SIZE, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

mouse_expert_discriminator = nn.Sequential(
    nn.Linear(DATA_SIZE, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

def genorate_expected_output(batch_size: int, discriminator_output_size: int, label: int):
    return torch.full((batch_size, discriminator_output_size), label, dtype=torch.float32) 

def genorate_normal_noise(batch_size: int, data_size: int):
    return torch.randn((batch_size, data_size))

def train_discriminator_loop(train_data: CellCensusDataLoader, discriminator: nn.Sequential, optimizer: torch.optim.Optimizer, loss_fn):
    discriminator.train()

    for batch, (batch_data, data_title) in enumerate(train_data):
        current_data: torch.Tensor
        expected_out: torch.Tensor

        if random.random() <= 0.5:
            current_data = batch_data
            expected_out = genorate_expected_output(64, 1, REAL_DATA)
        else:
            current_data = genorate_normal_noise(64, DATA_SIZE)
            expected_out = genorate_expected_output(64, 1, FAKE_DATA)

        pred = discriminator(current_data)
        loss = loss_fn(pred, expected_out)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            print(f'loss {loss:>7f}')

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(human_expert_discriminator.parameters(), lr=0.001)

train_discriminator_loop(human_data, human_expert_discriminator, optimizer, loss_fn)
