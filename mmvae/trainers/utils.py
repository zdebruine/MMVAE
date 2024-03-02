import torch
import matplotlib.pyplot as plt
import io
import PIL.Image
import numpy as np
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_curve, auc

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

def get_roc_stats(model, dataloader):
    """
    Computes the Receiver Operating Characteristic (ROC) statistics for a given model
    and dataloader at a specific epoch. It evaluates the model's ability to distinguish
    between real and fake data, which is especially relevant for models like discriminators
    in Generative Adversarial Networks (GANs).

    Parameters:
    - model: The model being evaluated. This function expects the model to have a method
             called `expert.discriminator` that can discriminate between real and fake data.
    - dataloader: A DataLoader instance that provides batches of real data for evaluation.
                  The function assumes that each batch from the dataloader can be directly
                  fed into the model.

    Returns:
    - fpr: An array containing the False Positive Rates computed at different thresholds.
    - tpr: An array containing the True Positive Rates computed at different thresholds.
    - roc_auc: The Area Under the Curve (AUC) of the ROC, summarizing the overall
               performance of the model in distinguishing between real and fake data.
    """
    real_scores = []
    fake_scores = []

    with torch.no_grad():
        for data in dataloader:
            real_scores.extend(model.expert.discriminator(data).view(-1).tolist())
            fake, _, _ = model(data)
            fake_scores.extend(model.expert.discriminator(fake).view(-1).tolist())
    
    scores = np.array(real_scores + fake_scores)
    y_true = np.array([1] * len(real_scores) + [0] * len(fake_scores))
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc
