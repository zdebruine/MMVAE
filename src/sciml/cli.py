
from lightning.pytorch.cli import LightningCLI
import torch

import lightning.pytorch.loggers.mlflow

class SCIMLCli(LightningCLI):
    
    def __init__(self, **kwargs):
        
        if not 'parser_kwargs' in kwargs:
            kwargs['parser_kwargs'] = {
                "default_env": True, 
            }
            
        from sciml import VAE, CellxgeneDataModule
        super().__init__(
            model_class=VAE, 
            datamodule_class=CellxgeneDataModule, 
            subclass_mode_data=True, 
            subclass_mode_model=True,
            **kwargs)
    
    def add_arguments_to_parser(self, parser):
        
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR)
        
if __name__ == "__main__":
    
    cli = SCIMLCli()
    
    print(cli.model)