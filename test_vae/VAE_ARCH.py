import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from mmvae.data import MappedCellCensusDataLoader


class VAE(nn.Module):
    def __init__(self, latent_size=128):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60664, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 256),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 60664),
            nn.LeakyReLU()
    
        )
        self.writer = tb.SummaryWriter()

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class VAETrainer:
    def __init__(self, model, train_loader, device, learning_rate=0.001, num_epochs=10, start_kl=0.0, end_kl=0.5, annealing_start=0, annealing_steps=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.start_kl = start_kl
        self.end_kl = end_kl
        self.annealing_start = annealing_start
        self.annealing_steps = annealing_steps

    def loss_function(self, recon_x, x: torch.Tensor, mu, logvar):
        reconstruction_loss = F.mse_loss(recon_x, x.to_dense(), reduction='sum')
        kl_divergence = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)
        return reconstruction_loss, kl_divergence

    def train(self):
        print("Start Training ....")
        for epoch in range(self.num_epochs):
            for i,x in enumerate(self.train_loader):
                x = x.to(self.device)

                recon_batch, mu, logvar = self.model(x)
                self.optimizer.zero_grad()
                
                if epoch >= self.annealing_start:
                    annealing_ratio = min((epoch - self.annealing_start) / self.annealing_steps, 1.0)
                    annealing = self.start_kl + annealing_ratio * (self.end_kl - self.start_kl)
                else:
                    annealing = self.start_kl

                recon_loss, kl_loss = self.loss_function(recon_batch, x, mu, logvar)
                annealing_kl = kl_loss * annealing
                loss = recon_loss + annealing_kl

                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Loss/ReconstructionFromTrainingData', loss.item(), i)

                if (i + 1) % 1875 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, self.num_epochs, i + 1, len(self.train_loader), loss.item()))
            
def main(device):
    batch_size = 32
    print("Device:", device)
    data_loader = MappedCellCensusDataLoader(
            batch_size=batch_size,
            device=device,
            file_path='/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz',
            load_all=True
    )
    model = VAE()
    trainer = VAETrainer(model, data_loader, device)
    trainer.train()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
