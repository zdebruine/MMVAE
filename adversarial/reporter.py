import shutil
import json
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

# Inherits from SummaryWriter, so check
# https://pytorch.org/docs/stable/tensorboard.html for more features

class Reporter(SummaryWriter):
    def __init__(self, log_dir: str, snapshot_directory=None):
        self.log_dir = log_dir
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.file_folder = log_dir + '/' + current_time
        super().__init__(self.file_folder)

        if snapshot_directory is None:
            snapshot_directory = self.file_folder + '/snapshots'
        self.snapshot_directory = snapshot_directory
        self.snapshot_count = 0

        self.losses = {}
        self.zero_losses()

    # Save a copy of the tensorboard file.
    def copy_to_snapshot(self):
        shutil.copytree(self.file_folder, self.snapshot_directory + '/snapshot_' + str(self.snapshot_count), ignore=shutil.ignore_patterns(self.file_folder))
        self.snapshot_count += 1

    # Zero losses. Not doing so may cause a memory leak...
    def zero_losses(self):
        for loss in self.losses:
            self.losses[loss] = 0
    
    def accumulate_loss(self, name: str, value: float):
        self.losses[name] += value

    # Write the losses to tensoboard file
    # minibatch_size is used to calculate the average loss
    def write_losses(self, iteration, n = 1):
        for loss in self.losses:
            self.add_scalar(loss, self.losses[loss]/n, iteration)

    # Add a list of loss labels to the reporter
    def add_loss_labels(self, names: list[str]):
        for name in names:
            self.losses[name] = 0
    


# Exapmle usage:
# reporter = Reporter('./logs/vae_gan', './logs/snapshots')
# reporter.add_loss_labels(['MSE Loss', 'KL Loss',]) 

# for epoch in range(num_epochs):
    # for batch in batches:
                #Caluculate error and backprop
                # reporter.accumulate_loss('MSE Loss', mse_error_scalar)
                # reporter.accumulate_loss('KL Loss', kl_error_scalar)
    
    # reporter.write_losses(epoch, len(batches))
    # reporter.zero_losses()

# reporter.close()
