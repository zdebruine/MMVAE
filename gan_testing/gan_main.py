import torch
from mmvae.trainers import HumanVAE_gan
from discriminators import annealing

def main(device):
    batch_size = 32
    num_iterations = 200
    annealer = annealing.CyclicalAnnealer(0.4, 3.2, num_iterations, 5, 0.5)
    writer = torch.utils.tensorboard.SummaryWriter('./logs/annealer_test/')
    for i in range(num_iterations):
        writer.add_scalar('Annealing', annealer(i), i)

    # trainer = HumanVAE_gan.HumanVAETrainer(
    #     batch_size,
    #     device,
    #     log_dir="/active/debruinz_project/cardell_taylor/logs/VAE_GAN_2"
    # )

    # print("done")

    # trainer.train(epochs=1)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
