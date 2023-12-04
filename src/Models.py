import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim) -> None:
        super(Encoder, self).__init__()
        
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dims)])
        # if len(hidden_dims) > 1:
        #     for i in range(len(hidden_dims)-1):
        #         self.encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.mean = nn.Linear(hidden_dims, latent_dim)
        self.var = nn.Linear(hidden_dims, latent_dim)

    def forward(self, x):
        for layer in self.encoder_layers:
            #print("Encoder", x)
            x = torch.relu(layer(x))
        return self.mean(x), self.var(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim) -> None:
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dims)])
        # if len(hidden_dims) > 1:
        #     for i in range(len(hidden_dims)-1):
        #         self.decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.output = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        for layer in self.decoder_layers:
            #print("Decoder", x)
            x = torch.relu(layer(x))
        return torch.relu(self.output(x))

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder) -> None:
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterize(self, mean, var):
        #print("Mean", mean, "Var", var)
        eps = torch.randn_like(var)#.to(DEVICE)
        return mean + var*eps

    def forward(self, x):
        mean, logvar = self.Encoder(x)
        z = self.reparameterize(mean, torch.exp(0.5 * logvar))
        x_hat = self.Decoder(z)
        return x_hat, mean, logvar









"""
def loss_function(x, x_hat, mean, logvar):
    #reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x.to_dense(), reduction='sum')
    #reconstruction_loss = nn.functional.mse_loss(x_hat, x.to_dense())
    reconstruction_loss = nn.functional.cross_entropy(x_hat, x.to_dense(), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KLD



def collate_fn(batch):
    #Custom collate function to return a list of tensors as a batch rather than one big tensor
    return batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(DEVICE)
BATCH_SIZE = 100
dataset = CellxGeneDataset(BATCH_SIZE)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print("Building Encoder...")
encode = Encoder(60664, 512, 128)
print("Success!\nBuilding Decoder...")
decode = Decoder(128, 512, 60664)
print("Success!\nBuilding model...")

test = VAE(encode, decode).to(DEVICE)
print("Success!")
optimizer = Adam(test.parameters())
#print("Wrapping for multi-GPU")
#test = torch.nn.DataParallel(test, device_ids=[0,1])
#test.to(DEVICE)
#print("Scucess!")
test.train()
for epoch in range(30):
    epoch_time = time.time()
    total_loss = 0
    for batch_idx, x in enumerate(dataset):
        train_loss = 0
        # print(f'Batch: {batch_idx}, Epoch: {epoch}')
        optimizer.zero_grad()
        x = x.to(DEVICE)
        # x = x.transpose(0, 1)

        x_hat, mean, log_var = test(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        train_loss += loss.item()
        total_loss += train_loss
        
        loss.backward()
        optimizer.step()
        #print("Batch", batch_idx+1, "Loss:", train_loss / x.shape[0])
    print("Time to run:", time.time() - epoch_time)
    print("Epoch", epoch + 1, "complete!", "Average Loss: ", train_loss / (batch_idx*BATCH_SIZE))
"""
