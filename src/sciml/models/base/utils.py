from typing import Literal, Union, Iterable
import torch


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor):
    """KL Divergence using mean and log_var layers on a Normal distribution"""
    # KL divergence term for each sample
    kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    # Sum over all dimensions for each sample, then mean over batch
    return torch.sum(kl_div, dim=1).mean()

def tag_log_dict(
    log_dict: dict[str, torch.Tensor], 
    tags: Iterable[str] = [], 
    sep: str = "/", 
    key_pos: Union[Literal['first'], Literal['last']] = 'first',
) -> dict[str, torch.Tensor]:
    """
    Annotates loss output keys with specified tags.

    Args:
        log_dict (Dict[str, torch.Tensor]): A dictionary containing loss outputs.
        tags (str): Tags to append or prepend to the keys.
        sep (str): Separator used to concatenate tags and keys.
        key_first (bool): If True, places the key before the tags. If False, places the tags before the key.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with updated keys based on the tags and separator.
    """
    
    tags_str = sep.join(tags)

    def key_generator(key):
        if key_pos == 'first':
            return f"{key}{sep}{tags_str}"
        if key_pos == 'last':
            return f"{tags_str}{sep}{key}"
    
    return {
        key_generator(key): value
        for key, value in log_dict.items()
    }