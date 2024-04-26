from typing import Tuple, Union, Any
import torch
import torch.nn as nn
import mmvae.models.utils as utils

class Discriminator(nn.Module):
    """
    A discriminator module for a generative adversarial network, with options for label smoothing and data shuffling.

    Parameters:
        discriminator (nn.Sequential): The neural network model that performs the discrimination task.
        shuffle (bool, optional): If True, shuffles the combined real and fake data every training step. Defaults to True.
        g_pretrain_steps (int, optional): Number of pre-training steps for the generator before training starts. Defaults to 0.
        alpha (float, optional): Smoothing factor for the real labels as described in https://arxiv.org/pdf/1606.03498. Defaults to 0.

    Attributes:
        discriminator (nn.Sequential): The underlying discriminator neural network.
        shuffle (bool): Flag to determine whether to shuffle data during training.
        g_pretrain_steps (int): Number of pre-training steps to perform.
        alpha (float): Smoothing parameter for real labels.
        _pretrain_step (int): Internal counter to track pre-training steps.
        _forward_modes (dict): Maps mode strings to their corresponding internal method calls.
    """
    def __init__(self, discriminator: nn.Sequential, shuffle: bool = True, 
                 g_pretrain_steps: int = 0, alpha: float = 0) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.shuffle = shuffle
        self.g_pretrain_steps = g_pretrain_steps
        self.alpha = alpha
        self._pretrain_step = 0

        self._forward_modes = {
            'train_d': self._train_d,
            'train_g': self._train_g,
            'test': self._test,
        }

    def _concatonate_data(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates and optionally shuffles real and fake data for discriminator input.

        Parameters:
            real_data (torch.Tensor): The real data samples.
            fake_data (torch.Tensor): The generated (fake) data samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the concatenated data and corresponding labels, both possibly shuffled.
        """
        real_labels = torch.ones((real_data.size(0), 1), 
                            device=real_data.device, dtype=torch.float32) - self.alpha
        fake_labels = torch.zeros((fake_data.size(0), 1), 
                            device=fake_data.device, dtype=torch.float32)

        combined_data = torch.cat([real_data.to_dense(), fake_data], dim=0)
        combined_labels = torch.cat([real_labels, fake_labels], dim=0)

        if self.shuffle:
            shuffled_indices = torch.randperm(combined_data.size(0))
            combined_data = combined_data[shuffled_indices]
            combined_labels = combined_labels[shuffled_indices]

        return combined_data, combined_labels

    def _test(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tests the discriminator with a set of real and fake data.

        Parameters:
            real_data (torch.Tensor): The real data samples.
            fake_data (torch.Tensor): The generated (fake) data samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Discriminator output and labels for the combined data set.
        """
        combined_data, combined_labels = self._concatonate_data(real_data, fake_data)
        return self.discriminator(combined_data), combined_labels

    def _train_d(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the discriminator on a mix of real and fake data.

        Parameters:
            real_data (torch.Tensor): The real data samples.
            fake_data (torch.Tensor): The generated (fake) data samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Discriminator output and labels for the combined data set.
        """
        if self._pretrain_step < self.g_pretrain_steps:
            return torch.zeros((1), requires_grad=True), torch.zeros((1))

        self._pretrain_step = 0
        combined_data, combined_labels = self._concatonate_data(real_data, fake_data)
        return self.discriminator(combined_data), combined_labels

    def _train_g(self, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trains the generator part of the GAN.

        Parameters:
            x_har (torch.Tensor): The generated (fake) data samples.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Generator output and training status, currently as placeholders.
        """
        self._pretrain_step += 1
        return self.discriminator(x_hat), torch.ones((x_hat.size(0), 1), 
                                                device=x_hat.device, dtype=torch.float32) - self.alpha

    def forward(self, mode: str, *args) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass routing to different operations based on mode.

        Parameters:
            mode (str): Operating mode ('train_g', 'train_d', 'test').
            *args: Additional arguments depending on the mode.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Result from the discriminator based on the chosen mode.

        Raises:
            ValueError: If an invalid mode is provided or necessary arguments are missing.
        """
        if mode == 'train_g':
            return self._train_g(args[0])
        elif mode in {'train_d', 'test'}:
            if len(args) != 2:
                raise ValueError("train_d and test mode require both real_data and fake_data.")
            return getattr(self, f"_{mode}")(*args)
        else:
            raise ValueError(f"{mode} is not a valid mode.")

class Expert(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator = None):
        super(Expert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        raise NotImplementedError()

class VAE(nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.var = var
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x: torch.Tensor) -> Tuple[Union[Any, torch.Tensor], Any, Any, Tuple[Any]]:
        
        x = self.encoder(x)
        mu = self.mean(x) 
        log_var = self.var(x)
        
        x = self.reparameterize(mu, log_var)
        x = self.decoder(x)
        return x, mu, log_var
