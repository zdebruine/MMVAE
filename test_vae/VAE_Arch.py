import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import math

class VAE(nn.Module):
    def __init__(self, latent_size=10):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten input to 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Adjust output dimension as needed
        )
        self.fc_mu = nn.Linear(32, latent_size)
        self.fc_var = nn.Linear(32, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )

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
        recon_x = recon_x.view(-1,1,28,28)
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
        self.mse_losses = []
        self.kl_losses = []
        self.kl_epoch = []
        self.annealing_values = []
        self.losses = []

    def loss_function(self, recon_x, x, mu, logvar):
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)
        return reconstruction_loss, kl_divergence

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (x, _) in enumerate(self.train_loader):
                x = x.to(self.device)

                recon_batch, mu, logvar = self.model(x)
                self.optimizer.zero_grad()
                
                if epoch >= self.annealing_start:
                    annealing_ratio = self.sigmoid(((epoch - self.annealing_start) / self.annealing_steps) * self.annealing_steps)
                    annealing = self.start_kl + annealing_ratio * (self.end_kl - self.start_kl)
                else:
                    annealing = self.start_kl

                recon_loss, kl_loss = self.loss_function(recon_batch, x, mu, logvar)
                annealing_kl = kl_loss * annealing
                loss = recon_loss + annealing_kl

                loss.backward()
                self.optimizer.step()

                self.losses.append(loss.item())
                self.kl_losses.append(annealing_kl.item())
                self.mse_losses.append(recon_loss.item())

                if (i + 1) % 1875 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, self.num_epochs, i + 1, len(self.train_loader), loss.item()))
            
            self.kl_epoch.append(annealing_kl.item())
            self.annealing_values.append(annealing)
    
    def graphs(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(self.train_loader):
                x = x.to(self.device)

                x_hat, _, _ = self.model(x)

                break
        #smooth loss graph
        smoothed_total_losses = gaussian_filter1d(self.losses, sigma=2)
        smoothed_mse_losses= gaussian_filter1d(self.mse_losses, sigma=2)
        #Graphing Area
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)

        #plot training loss
        axes[0, 0].plot(smoothed_total_losses, label='Training Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        #axes[0, 0].set_ylim(0, 0.1)
        axes[0, 0].set_title("Training Loss, LR={}".format(self.lr))
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        #plot the annealing schedule
        axes[0, 1].plot(range(1, self.num_epochs + 1), self.annealing_values, marker='o', linestyle='-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Annealing Coefficient')
        axes[0, 1].set_title('Annealing Schedule for KL Divergence Loss')
        axes[0, 0].legend()
        axes[0, 1].grid(True)

        #plot kl loss vs mse loss
        axes[1, 0].plot(self.kl_losses, label='KL_Losses', color = 'red')
        axes[1, 0].plot(smoothed_mse_losses, label='MSE_Losses', color = 'blue')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Red->KL : Blue->Loss')
        axes[1, 0].set_ylim(0,1500)
        axes[1, 0].set_title("Kl Loss vs Recon Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        #plot kl loss over epochs
        axes[1,1].plot(range(1, self.num_epochs + 1), self.kl_epoch, marker='o', linestyle='-')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel("KL Loss")
        axes[1, 1].set_title('KL Loss over Epochs')
        axes[1, 1].grid(True)

         # make gap smaller 
        fig, axes = plt.subplots(nrows=15, ncols=4, figsize=(10, 60))
        plt.subplots_adjust(hspace=0.5)

        axes[1, 0].set_title('Original Images')
        axes[1, 1].set_title('Reconstructed Images')
        axes[1, 2].set_title('Original Images')
        axes[1, 3].set_title('Reconstructed Images')

        for i in range(15):

            original_image = x[i].view(28, 28).cpu().numpy()
            reconstructed_image = x_hat[i].view(28, 28).cpu().numpy()
            
            axes[i, 0].imshow(original_image)
            #axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(reconstructed_image)
            #axes[i, 1].set_title('Reconstructed Image')
            axes[i, 1].axis('off')
            j = i*2+1
            original_image = x[j].view(28, 28).cpu().numpy()
            reconstructed_image = x_hat[j].view(28, 28).cpu().numpy()

            # Add your own images or modify as per your requirement
            axes[i, 2].imshow(original_image)
            #axes[i, 2].set_title('Original Image')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(reconstructed_image)
            #axes[i, 3].set_title('Reconstructed Image')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.show()
        
def main():
    batch_size = 32
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)) 
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True)

    model = VAE()
    trainer = VAETrainer(model, train_loader, device)
    trainer.train()
    trainer.graphs()

if __name__ == "__main__":
    main()
