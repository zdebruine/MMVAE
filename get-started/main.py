import torch
from mmvae.trainers import HumanVAETrainer
from datetime import datetime

def main(device):
    # Define any hyperparameters
    batch_size = 512
    
    # Create trainer instance
    trainer = HumanVAETrainer(
        batch_size, 
        device,
        log_dir='/home/howlanjo/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "_JUST_10_TRY_LOADING",
        snapshot_path="/home/howlanjo/dev/MMVAE/snapshots/model" ,
    )
    # Train model with number of epochs
    trainer.train(epochs = 1, load_snapshot=False)
    
if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
