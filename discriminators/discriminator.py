import torch
import random
from torch import float32, nn, real
from torch.cuda import is_available
from d_mmvae.data import CellCensusDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f'Using {device}')

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
    return torch.full((batch_size, discriminator_output_size), label, dtype=torch.float32, device=device)

def genorate_normal_noise(batch_size: int, data_size: int):
    return torch.randn((batch_size, data_size), device=device)

def test_loop(test_data: CellCensusDataLoader, discriminator: nn.Sequential, loss_fn):
    discriminator.eval()
    test_loss, real_correct, fake_correct = 0, 0, 0

    size = len(test_data)*64

    with torch.no_grad():
        for batch_data, data_title in test_data:
            batch_size = len(batch_data)
            real_pred = discriminator(batch_data)
            fake_pred = discriminator(genorate_normal_noise(batch_size, DATA_SIZE))
            test_loss += loss_fn(real_pred, genorate_expected_output(batch_size, 1, REAL_DATA)) + loss_fn(fake_pred, genorate_expected_output(batch_size, 1, FAKE_DATA))

            if real_pred.item() >= 0.5:
                real_correct += 1

            if fake_pred.item() < 0.5:
                fake_correct += 1

    test_loss /= len(test_data)
    real_correct /= size
    fake_correct /= size

    print(f'Test Error: {test_loss}, fake data accuracy: {100*fake_correct}, real data accuracy: {100*real_correct}\n')


def train_discriminator_loop(train_data: CellCensusDataLoader, discriminator: nn.Sequential, optimizer: torch.optim.Optimizer, loss_fn):
    discriminator.to(device)
    discriminator.train()

    for batch, (batch_data, data_title) in enumerate(train_data):
        batch_size = len(batch_data)
        batch_data = batch_data.to(device)
        real_expected_out = genorate_expected_output(batch_size, 1, REAL_DATA)
        fake_expected_out = genorate_expected_output(batch_size, 1, FAKE_DATA)

        real_pred = discriminator(batch_data)
        real_loss = loss_fn(real_pred, real_expected_out)
        real_loss.backward()

        fake_input = genorate_normal_noise(batch_size, DATA_SIZE)
        fake_pred = discriminator(fake_input)
        fake_loss = loss_fn(fake_pred, fake_expected_out)
        fake_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = real_loss.item() + fake_loss.item()
            print(f'loss {loss:>7f}, batch #: {batch}')

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(human_expert_discriminator.parameters(), lr=0.001)

train_discriminator_loop(human_data, human_expert_discriminator, optimizer, loss_fn)
test_loop(human_data, human_expert_discriminator, loss_fn)
