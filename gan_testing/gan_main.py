import torch
from mmvae.trainers import HumanVAE_gan
from datetime import datetime

#discrim_div = [100, 1000, 5000, 20000, 50000, 100000]
discrim_div = [100000]

def main(device):
    time = datetime.now()
    formatted = time.strftime("%Y-%m-%d_%H-%M-%S")

    batch_size = 32

    for div in discrim_div:
        trainer = HumanVAE_gan.HumanVAETrainer(
            batch_size=batch_size,
            lr=0.0001,
            annealing_steps=50,
            discriminator_div=div,
            device=device,
            log_dir=f"/active/debruinz_project/jack_lukomski/logs/{formatted}"
        )
        trainer.train(epochs=2)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
