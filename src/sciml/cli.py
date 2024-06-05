
from pathlib import Path
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
import torch

from lightning.pytorch.loggers import MLFlowLogger

class CustomMLFLowLogger(MLFlowLogger):
    
    @property
    def save_dir(self):
        save_dir = Path(super().save_dir)
        return save_dir.joinpath(self.run_id)
        

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
            save_config_callback=SaveConfigCallback,
            save_config_kwargs={
                "config_filename": "configs/config.yaml"
            },
            **kwargs)
    
    def add_arguments_to_parser(self, parser):
        
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR)
        
        
if __name__ == "__main__":
    
    cli = SCIMLCli()
    
    print(cli.model)