import torch
import torch.nn as nn
import mmvae.models.utils as utils
import numpy as np
from mmvae.data import MappedCellCensusDataLoader
import time
import datetime
import sys

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60664, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 0.8),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768, 0.8),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256, 0.8)
        )
        
        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768, 0.8),
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, 0.8),
            nn.Linear(1024, 60664),
            nn.Sigmoid(),
        )

        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)
        utils._submodules_init_weights_xavier_uniform_(self.fc_mu)
        utils._xavier_uniform_(self.fc_var, -1.0)


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
        return recon_x
    

input_file = sys.argv[1]

#Create Instance of Device and Data Loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader =  MappedCellCensusDataLoader(
    batch_size=512,
    device=device,
    file_path= input_file,  #'/actvive/debruinz_project/CellCensus_3M/3m_human_chunk_30.npz',
    load_all=True,
    shuffle=False
)

#Load Save Model 
model_path = '/active/debruinz_project/parker_bernreuter/Trained_Model.pth'
model_dict = torch.load(model_path, map_location=torch.device(device))

#Create Instance of VAE and load to device
model = VAE()
model.to(device)

#Set to Evaluation Mode
model.load_state_dict(model_dict)
model.eval()

#Start the timer
start_time = time.time()


all_outputs = []

with torch.no_grad():
    for i,x in enumerate(loader):
        x = x.to(device)
        output = model(x)
        output = output.cpu().numpy()
        for z in output:
            all_outputs.append(z)


dt_object = datetime.datetime.fromtimestamp(start_time)
format_time = dt_object.strftime('%Y-%m-%d-%H-%M-%S')

np.savez('/active/debruinz_project/parker_bernreuter/model_outputs/model_output_'+ str(format_time) +'.npz', all_outputs)