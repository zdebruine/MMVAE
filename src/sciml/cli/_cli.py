import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

class SCIMLCli(LightningCLI):
    
    def __init__(self, **kwargs):
        
        if not 'parser_kwargs' in kwargs:
            kwargs['parser_kwargs'] = {
                "default_env": True, 
                "parser_mode": "omegaconf"
            }
            
        from sciml import VAEModel, CellxgeneDataModule
        super().__init__(
            model_class=VAEModel, 
            datamodule_class=CellxgeneDataModule, 
            subclass_mode_data=True, 
            subclass_mode_model=True,
            **kwargs)
    
    def add_arguments_to_parser(self, parser):
        
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR)
        
        parser.link_arguments('data.init_args.batch_size', 'model.init_args.batch_size')