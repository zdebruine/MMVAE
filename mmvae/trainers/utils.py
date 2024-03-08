import torch


def kl_divergence(mu, logvar, reduction="sum"):
    """
    Calculate the KL divergence between a given Gaussian distribution q(z|x)
    and the standard Gaussian distribution p(z).

    Parameters:
    - mu (torch.Tensor): The mean of the Gaussian distribution q(z|x).
    - sigma (torch.Tensor): The standard deviation of the Gaussian distribution q(z|x).

    Returns:
    - torch.Tensor: The KL divergence.
    """
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if reduction == "mean":
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
