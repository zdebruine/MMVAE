import torch

def kl_divergence(mu, logvar):
    """
    Calculate the KL divergence between a given Gaussian distribution q(z|x)
    and the standard Gaussian distribution p(z).

    Parameters:
    - mu (torch.Tensor): The mean of the Gaussian distribution q(z|x).
    - sigma (torch.Tensor): The standard deviation of the Gaussian distribution q(z|x).

    Returns:
    - torch.Tensor: The KL divergence.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
