import torch
from mmvae.trainers.Arch_Trainer import VAETrainer

def main():
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    trainer = VAETrainer(device)
    trainer.train()

if __name__ == "__main__":
    main()
