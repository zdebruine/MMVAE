import torch
import torch.nn as nn
import torch.optim as optim
from discriminators.visualize import rocplot

from torch.utils.tensorboard import SummaryWriter

class MetaDiscriminator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MetaDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def meta_discriminator_test(generator: nn.Module, data_loader, writer: SummaryWriter, trainin_epochs=1, lr=1e-14):
    discriminator = MetaDiscriminator(60664, 512, 1)
    discriminator.to('cuda')
    optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(trainin_epochs):
        for i, train_data in enumerate(data_loader):
            optimizer.zero_grad()

            real_pred = discriminator(train_data)
            real_loss = nn.L1Loss()(real_pred, torch.ones_like(real_pred))
            real_loss.backward()
            # gen fake data
            fake_data, _, _ = generator(train_data)

            # train discriminator with fake data
            fake_pred = discriminator(fake_data)
            fake_loss = nn.L1Loss()(fake_pred, torch.zeros_like(fake_pred))
            fake_loss.backward()

            optimizer.step()

    discriminator.eval()
    rocplot.roc_plot(discriminator, generator, data_loader, torch.device('cuda'), writer) 

