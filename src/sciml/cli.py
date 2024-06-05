
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
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
            
        if not 'save_config_kwargs' in kwargs:
            kwargs['save_config_kwargs'] = {
                "overwrite": True
            }
            
        from sciml import VAE, CellxgeneDataModule
        super().__init__(
            model_class=VAE, 
            datamodule_class=CellxgeneDataModule, 
            subclass_mode_data=True, 
            subclass_mode_model=True,
            **kwargs)
    
    def add_arguments_to_parser(self, parser):
        
        parser.add_argument('--root_dir', type=str, help="Root Directory for experiment tracking")
        
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR)
        
    def before_instantiate_classes(self) -> None:
        print(self.config)
        exit(1)
        
if __name__ == "__main__":
    
    cli = SCIMLCli()
    
    print(cli.model)