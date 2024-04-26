import torch
from mmvae.finetuners import HumanVAE_FineTune
from datetime import datetime
import torch.nn as nn

SNAPSHOT_EXPERT_PATH = '/home/howlanjo/dev/MMVAE/snapshots/model_expert'
SNAPSHOT_SHARED_PATH = '/home/howlanjo/dev/MMVAE/snapshots/model_shared'

def load_snapshot(path) -> tuple[nn.Module]:
    snapshot = torch.load(path)
    return snapshot["MODEL_STATE"]

def main(device):
    # Define any hyperparameters
    batch_size = 512
    
    #Loading snapshot
    expert = load_snapshot(SNAPSHOT_EXPERT_PATH)
    shared_encoder = load_snapshot(SNAPSHOT_SHARED_PATH+'_encoder')
    shared_decoder = load_snapshot(SNAPSHOT_SHARED_PATH+'_decoder')
    shared_var = load_snapshot(SNAPSHOT_SHARED_PATH+'_var')
    shared_mean = load_snapshot(SNAPSHOT_SHARED_PATH+'_mean')
    
    
    # Create trainer instance
    ft_trainer = HumanVAE_FineTune(
        batch_size, 
        device,
        log_dir='/home/howlanjo/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "FINE_TUNE_STALE",
        snapshot_path="/home/howlanjo/dev/MMVAE/snapshots/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "FINE_TUNE_STALE" ,
    )
    
    ft_trainer.model.expert.load_state_dict(expert)
    ft_trainer.model.shared_vae.encoder.load_state_dict(shared_encoder)
    ft_trainer.model.shared_vae.decoder.load_state_dict(shared_decoder)
    ft_trainer.model.shared_vae.var.load_state_dict(shared_var)
    ft_trainer.model.shared_vae.mean.load_state_dict(shared_mean)
    
    # Freeze parameters of encoder and decoder
    for param in ft_trainer.model.expert.parameters():
        param.requires_grad = False
    for param in ft_trainer.model.shared_vae.encoder.parameters():
        param.requires_grad = False
    for param in ft_trainer.model.shared_vae.decoder.parameters():
        param.requires_grad = False
    
    # Train model with number of epochs
    ft_trainer.train(epochs = 1)
        
if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
