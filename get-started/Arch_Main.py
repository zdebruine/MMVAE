import torch
from mmvae.trainers.Arch_Trainer import VAETrainer
from mmvae.data import MappedCellCensusDataLoader

def main():
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # data_loader = MappedCellCensusDataLoader(
    #         batch_size=batch_size,
    #         device=device,
    #         file_path='/active/debruinz_project/CellCensus_3M_Full/3m_human_full.npz',
    #         #3m_mouse_chunk_10.npz
    #         load_all=True
    # )
    trainer = VAETrainer(device)
    trainer.train()

if __name__ == "__main__":
    main()
