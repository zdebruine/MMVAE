import torch
from Dataset import CellxGeneDataset
from torch.utils.data import DataLoader
from Models import Encoder, Decoder, VAE
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: CellxGeneDataset,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        dataset
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.dataset = dataset
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    def _run_batch(self, source):
        self.optimizer.zero_grad()
        output, mean, logvar = self.model(source)
        recon_loss = torch.nn.CrossEntropyLoss()(output, source.to_dense())
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + KLD
        loss.backward()
        self.optimizer.step()
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for i in range(0, 100):
            chunk_time = time.time()
            self.train_data.sampler.set_epoch(epoch+i)
            for source in self.train_data:
                source = source.to(self.local_rank)
                self._run_batch(source)
            print(f"Chunk processed in: {time.time() - chunk_time} | Chunks loaded: {self.dataset.chunks_loaded}")
    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            epoch_time = time.time()
            self._run_epoch(epoch)
            print(f"Epoch ran in: {time.time() - epoch_time}")
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
def ddp_setup() -> None:
    init_process_group(backend="nccl")
def load_train_objs(batch_size):
    train_set = CellxGeneDataset(batch_size)  # load your dataset
    model = VAE(Encoder(60664, 512, 128), Decoder(128, 512, 60664))  # load your model
    optimizer = torch.optim.Adam(model.parameters())
    return train_set, model, optimizer
def prepare_dataloader(dataset: CellxGeneDataset):
    return DataLoader(
        dataset,
        batch_size=None,
        # pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
def main(total_epochs: int, save_every: int, batch_size: int, snapshot_path: str):
    ddp_setup()
    dataset, model, optimizer = load_train_objs(batch_size)
    train_data = prepare_dataloader(dataset)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, dataset)
    trainer.train(total_epochs)
    destroy_process_group()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('batch_size', type=int, help='Input batch size on each device')
    parser.add_argument('snapshot_path', type=str, help='Path to save/load training snapshots')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    main(args.total_epochs, args.save_every, args.batch_size, args.snapshot_path)
