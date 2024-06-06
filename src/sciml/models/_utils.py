import torch
import torch.nn as nn
import torch.nn.init as init

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor):
    # KL divergence term for each sample
    kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    # Sum over all dimensions for each sample, then mean over batch
    return torch.sum(kl_div, dim=1).mean()

def init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # TODO: Explore fan_out vs fan_in
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
            
def tag_loss_outputs(loss_outputs: dict[str, torch.Tensor], tag: str, sep="_"):
    outputs = {
        f"{tag}{sep}{key}": value
        for key, value in loss_outputs.items()
    }
    del loss_outputs
    return outputs