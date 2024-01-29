import torch
from d_mmvae.trainers import ExampleTrainer

def main(device):
    # Define any hyperparameters
    batch_size = 32
    # Create trainer instance
    trainer = ExampleTrainer(
        batch_size,
        device
    )
    # Train model with number of epochs
    trainer.train(epochs=1)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
