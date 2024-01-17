import torch
from d_mmvae import Models

class TrainerStorage:

    def __init__(self, snapshot_path: str, save_every: int):
        self._save_every = save_every
        self._snapshot_path = snapshot_path

    def load_snapshot(self) -> tuple[int, torch.nn.Module]:
        snapshot = torch.load(self._snapshot_path)
        return snapshot["MODEL_STATE"], snapshot["EPOCHS_RUN"]
    
    def save_snapshot(self, model: torch.nn.Module, epoch: int) -> None:
        snapshot = {}
        snapshot["MODEL_STATE"] = model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self._snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self._snapshot_path}")


class DDMVAETrainer:

    def __init__(self, model: Models.MMVAE, loaders, optimizers, storage: TrainerStorage):
        self.model = model
        self.storage = storage

    def train(self, epochs, load_snapshot=False):
        if load_snapshot:
            self.storage.load_snapshot()

        for epoch in range(epochs):
            self._train_epoch(epoch)
            if (epoch + 1) % self.storage._save_every == 0:
                self.storage.save_snapshot(self.models)

    def _train_epoch(self, epoch):
        self.model.train()

        output, mean, logvar = self.model(source)
        recon_loss = torch.nn.CrossEntropyLoss()(output, source.to_dense())
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + KLD
        loss.backward()
        self.optimizer.step()