import torch
import d_mmvae.models as M
from d_mmvae.trainers import MMVAETrainer as Trainer
from d_mmvae.data import MultiModalLoader, CellCensusDataLoader
from torch import nn
from d_mmvae.models.MMVAE import MMVAE, SharedDecoder, SharedEncoder

def configure_dataloader(batch_size: int):
    expert1 = CellCensusDataLoader('expert1', directory_path="/active/debruinz_project/tony_boos/csr_chunks", masks=['chunk*'], batch_size=batch_size, num_workers=2)
    expert2 = CellCensusDataLoader('expert2', directory_path="/active/debruinz_project/tony_boos/csr_chunks", masks=['chunk*'], batch_size=batch_size, num_workers=2)
    return MultiModalLoader(expert1, expert2)

def configure_model():
    num_experts = 2
    return MMVAE(
        nn.ModuleDict({
            'expert1': M.Expert(
                nn.Sequential(
                        nn.Linear(60664, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 60664),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Linear(60664, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )),
            'expert2': M.Expert(
                nn.Sequential(
                        nn.Linear(60664, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 60664),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Linear(60664, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                ))
        }),
        M.VAE(
            SharedEncoder(num_experts),
            SharedDecoder(),
            nn.Linear(64, 64),
            nn.Linear(64, 64)
        )
    )

def configure_optimizers(model: MMVAE):
    optimizers = {}
    for name in model.experts.keys():
        expert = model.experts[name]
        optimizers[f'{name}-enc'] = torch.optim.Adam(expert.encoder.parameters())
        optimizers[f'{name}-dec'] = torch.optim.Adam(expert.decoder.parameters())
        optimizers[f'{name}-disc'] = torch.optim.Adam(expert.discriminator.parameters())

    optimizers['shr_enc_disc'] = torch.optim.Adam(model.shared_vae.encoder.discriminator.parameters())
    optimizers['shr_vae'] = torch.optim.Adam(list(model.shared_vae.encoder.parameters()) + list(model.shared_vae.decoder.parameters()))
    return optimizers

def main(device):
    dataloader = configure_dataloader(batch_size=32)
    model = configure_model()
    optimizers = configure_optimizers(model)

    trainer = Trainer(
        model,
        optimizers,
        dataloader,
        device
    )

    trainer.train(1)


if __name__ == "__main__":

    CUDA = False

    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"

    main(device)
