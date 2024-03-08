import torch
from mmvae.trainers import ExampleTrainer

def main(device):
    # Define any hyperparameters
    batch_size = 32
    # Create trainer instance
    trainer = ExampleTrainer(
        batch_size,
        device,
        log_dir="/path/to/your/log/directory"
    )
    # Train model with number of epochs
    trainer.train(epochs=1)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
