from lightning.pytorch.cli import LightningCLI
import torch

class SCIMLCli(LightningCLI):
    
    def __init__(self, *args, **kwargs):
        if not 'parser_kwargs' in kwargs:
            kwargs['parser_kwargs'] = {"default_env": True}
        super().__init__(*args, **kwargs)
    
    def add_arguments_to_parser(self, parser):
        
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR)
        
        parser.set_defaults({
            "trainer.logger": [
                {
                    "class_path": "lighting.pytorch.loggers.MLFlowLogger",
                    "init_args": {
                        "log_model": "all"
                    }
                }, 
                {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                }
            ]
        })

        
if __name__ == "__main__":

    cli = SCIMLCli()
    
    print(cli.model)