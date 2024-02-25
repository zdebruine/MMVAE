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

def get_batch_tpr_fpr(discriminator: torch.nn.Sequential, real_batch_data, fake_batch_data, real: int = 1, fake: int = 0):

    with torch.no_grad():
        x_real = discriminator(real_batch_data)
        x_fake = discriminator(fake_batch_data)

    TP = ((x_real >= 0.5).float()).sum().item()
    TN = ((x_fake < 0.5).float()).sum().item()
    FP = ((x_fake >= 0.5).float()).sum().item()
    FN = ((x_real < 0.5).float()).sum().item()

    TPR = (TP / (TP + FN))
    FPR = (FP / (FP + TN))

    return TPR, FPR
