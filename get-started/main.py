import torch
from mmvae.trainers import HumanVAE
import datetime

def main(device):
    # Define any hyperparameters
    batch_size = 512
    
    # Create trainer instance
    trainer = HumanVAE(
        batch_size, 
        device,
        log_dir='/home/howlanjo/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "_JUST_10",
        snapshot_path="/home/howlanjo/dev/MMVAE/snapshots/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_JUST_10" ,
    )
    # Train model with number of epochs
    trainer.train(epochs = 50)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
