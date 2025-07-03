import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

# Simple encoder with separate latent networks (mean/variance layers)
class LatentNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LatentNetwork, self).__init__()
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        q_m = self.fc_mean(x)
        # Compute the variance of the latent variables
        # and add epsilon for numerical stability
        q_v = torch.exp(self.fc_logvar(x)) + 1e-4

        # Create a normal distribution with the computed mean and variance
        dist = Normal(q_m, q_v.sqrt())

        # Sample the latent variables and apply the transformation
        latent = dist.rsample()

        return dist, latent

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, latent_dim)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

# Simple shared decoder
class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(SharedDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

# Reconstruction loss (mean squared error in this case)
def reconstruction_loss(reconstructed, target):
    return F.mse_loss(reconstructed, target, reduction='sum')

# Merging function for latent variables (simple concatenation)
def merge_latent(merge, latent1, latent2):
    merged_latent = torch.cat((latent1, latent2), dim=1)
    print(merged_latent)
    print(f"Merged latent grad_fn: {merged_latent.grad_fn}")
    return merge(merged_latent)

# Example network setup
input_dim = 4
latent_dim = 2
output_dim = 4

encoder = Encoder(input_dim, latent_dim)
latent_net_1 = LatentNetwork(latent_dim, latent_dim)
latent_net_2 = LatentNetwork(latent_dim, latent_dim)
merge = nn.Linear(latent_dim * 2, latent_dim)
decoder = SharedDecoder(latent_dim, output_dim)

# Optimizer
optimizer = optim.Adam(list(latent_net_1.parameters()) + list(latent_net_2.parameters()) + list(decoder.parameters()), lr=0.001)

# Dummy input and target
X = torch.rand(1, input_dim)  # Example input
target = torch.rand(1, output_dim)  # Example target

x = encoder(X)

# Forward pass through both latent networks
dist1, latent1 = latent_net_1(x)
dist2, latent2 = latent_net_2(x)

# Print grad_fn of mean/logvar
# print(f"dist 1 grad_fn: {dist1.grad_fn}")
print(f"latent 1 grad_fn: {latent1.grad_fn}")
# print(f"dist 2 grad_fn: {dist2.grad_fn}")
print(f"latent 2 grad_fn: {latent2.grad_fn}")

# Merge latent spaces and pass through the shared decoder
merged_latent = merge_latent(merge, latent1, latent2)
reconstructed = decoder(merged_latent)

# Print grad_fn of merged_latent and reconstructed output
print(f"Merged latent grad_fn: {merged_latent.grad_fn}")
print(f"Reconstructed output grad_fn: {reconstructed.grad_fn}")

# Compute losses
pz = Normal(torch.zeros_like(merged_latent), torch.ones_like(merged_latent))
kl_loss_1 = kl_divergence(dist1, pz).sum(dim=-1).mean()
kl_loss_2 = kl_divergence(dist2, pz).sum(dim=-1).mean()
reconstruction_loss_value = reconstruction_loss(reconstructed, target)

# Total loss (reconstruction loss + both KL losses)
total_loss = 1 * reconstruction_loss_value #+ kl_loss_1 #+ kl_loss_2

print("KL1", kl_loss_1)
print("KL2", kl_loss_2)
print("Recon", reconstruction_loss_value)
print("Loss", total_loss)

# Backpropagation
optimizer.zero_grad()
total_loss.backward()

# Print gradients of mean/logvar layers, decoder
print("\nGradients after backprop:")
print("Encoder Layer Gradient:\n", encoder.fc.weight.grad)
print("Latent Net 1 Mean Layer Gradient:\n", latent_net_1.fc_mean.weight.grad)
print("Latent Net 1 Logvar Layer Gradient:\n", latent_net_1.fc_logvar.weight.grad)
print("Latent Net 2 Mean Layer Gradient:\n", latent_net_2.fc_mean.weight.grad)
print("Latent Net 2 Logvar Layer Gradient:\n", latent_net_2.fc_logvar.weight.grad)
print("Merge Latent Layer Gradient:\n", merge.weight.grad)
print("Decoder Layer Gradient:\n", decoder.fc.weight.grad)

# Optimizer step
optimizer.step()
