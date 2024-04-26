import torch
import numpy as np
from mmvae.data import configure_singlechunk_dataloaders
import mmvae.models.Arch_Model as Arch_Model
import time
import datetime

#Create Instance of Device and Data Loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = configure_singlechunk_dataloaders(
            data_file_path = '/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz',#Path to Data being sent through model
            metadata_file_path = None,
            shuffle = False,
            train_ratio = 1,
            batch_size = 128,
            device = None
        )

#Load Save Model
model_path = '/active/debruinz_project/parker_bernreuter/Trained_Model_Full_04-10-24.pth'#Path to file with model
model_dict = torch.load(model_path, map_location=torch.device(device))

#Create Instance of VAE and load to device
model = Arch_Model.VAE()
model.to(device)

#Load weights and set to Evaluation Mode
model.load_state_dict(model_dict)
model.eval()

#Instantiate dense output matrix
all_outputs = []

#Loop to send data through model
with torch.no_grad():
    for i,(x, _) in enumerate(loader):
        #Load data to device and run 
        x = x.to(device)
        output = model(x)
        output = output[0].cpu().numpy().astype(np.float16) 
        for z in output:
            all_outputs.append(z)


#Save the output in file with time as name
dt_object = datetime.datetime.fromtimestamp(time.time())
format_time = dt_object.strftime('%Y-%m-%d-%H-%M-%S')
np.savez('/active/debruinz_project/parker_bernreuter/model_outputs/model_output_'+ str(format_time) +'.npz', all_outputs)