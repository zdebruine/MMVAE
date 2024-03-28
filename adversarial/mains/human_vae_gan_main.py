import torch
import json
from adversarial.trainers import human_vae_gan

HPARAMS_PATH = '/active/debruinz_project/jack_lukomski/MMVAE_Adversarial_Team/adversarial/hparams/human_vae_gan.json'

def main(device):

    with open(HPARAMS_PATH, mode='r') as file:
        json_hparams = json.load(file)

    hparams = human_vae_gan.HumanVAEConfig(config=json_hparams)
    trainer = human_vae_gan.HumanVAETrainer(device=device, hparams=hparams)

    trainer.train(100)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
