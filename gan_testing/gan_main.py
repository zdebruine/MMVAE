import torch
from mmvae.trainers import HumanVAE_gan
from discriminators import annealing

def main(device):
    batch_size = 32

    trainer = HumanVAE_gan.HumanVAETrainer(
         batch_size,
         device,
         log_dir="/active/debruinz_project/cardell_taylor/logs/VAE_GAN_2"
     )

    print("done")

    trainer.train(epochs=1)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
