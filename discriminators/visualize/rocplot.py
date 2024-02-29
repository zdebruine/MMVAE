import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader


"""
This function is used to plot the ROC curve for the discriminator model.
It takes in the discriminator model, generator model, dataloader and device as input.
It uses the discriminator model to predict the real and fake data and then uses the
true values and model predictions to calculate the ROC curve. It then plots the ROC curve
using tensorboard and saves it in the logs folder. 

True positive rate and false positive rate are scaled by 100 to make it easier to read in tensorboard.

Parameters:
    discriminator: nn.Module
    generator: nn.Module
    dataloader: DataLoader
    device: torch.device
"""
@torch.no_grad
def roc_plot(discriminator: nn.Module, generator: nn.Module, dataloader: DataLoader, device: torch.device, 
             writer: SummaryWriter):
    model_prediction = []
    true_value = []
    for i, data in enumerate(dataloader):
        data = data.to(device)
        # Generate fake data
        fake_data, _, _ = generator(data)
        # Predict fake data
        fake_pred = discriminator(fake_data).view(-1)

        # Predict real data
        real_pred = discriminator(data).view(-1)

        # append model predictions from both real data
        model_prediction.append(real_pred)
        # append model predictions from both fake data
        model_prediction.append(fake_pred)
        
        # append true values for real data
        true_value.append(torch.ones_like(real_pred).to('cpu').numpy())
        # append true values for fake data
        true_value.append(torch.zeros_like(fake_pred).to('cpu').numpy())

    # detach and convert to numpy
    model_prediction = torch.cat(model_prediction).to('cpu').numpy()
    true_value = np.concatenate(true_value)

    # calculate ROC curve
    false_positive_rate, true_positive_rate, _ = roc_curve(true_value, model_prediction)

    # scale the false positive rate by 100
    false_positive_rate = false_positive_rate * 100
    # scale the true positive rate by 100
    true_positive_rate = true_positive_rate * 100

    # calculate AUC
    # auc = roc_auc_score(true_value, model_prediction)

    # plot ROC curve with tensorboard and save it

    # writer = SummaryWriter('../../logs/roc/VAE_GAN_2')
    # uses step as the false positive rate
    for i in range(len(true_positive_rate)):
        writer.add_scalar('True_Positive_Rate', true_positive_rate[i], false_positive_rate[i])
