from typing import Any, Optional
import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from sciml.models import BaseVAEModel


class SCIMLCli(LightningCLI):
    """
        LightningCLI meant to ease in setting default argumetns and 
    logging parameters. All Models of subclass BaseVAEModel should use this
    cli.
    """
    def __init__(
        self, 
        extra_arguments: Optional[list[dict[str, Any]]] = None,
        extra_parser_kwargs: dict[str, Any] = {},
        **kwargs
    ):
        self._run = kwargs.get('run', True) 
        self.extra_args = extra_arguments
        
        super().__init__(
            model_class=BaseVAEModel, 
            subclass_mode_data=True, 
            subclass_mode_model=True,
            parser_kwargs={
                "default_env": True, 
                "parser_mode": "omegaconf",
                **extra_parser_kwargs
            },
            **kwargs)
    
    def add_arguments_to_parser(self, parser):
        # Add arguments for logging and ease of access
        parser.add_argument('--default_root_dir', required=True, help="Default root directory")
        parser.add_argument('--experiment_name', required=True, help="Name of experiment directory")
        parser.add_argument('--run_name', required=True, help="Name of the experiment run")
        
        if self._run == False:
            parser.add_argument('--ckpt_path', default="")
        
        # Link ease of access arguments to their respective targets
        parser.link_arguments('default_root_dir', 'trainer.default_root_dir')
        parser.link_arguments('default_root_dir', 'trainer.logger.init_args.save_dir')
        parser.link_arguments('experiment_name', 'trainer.logger.init_args.name')
        parser.link_arguments('run_name', 'trainer.logger.init_args.version')
        
        parser.link_arguments('data.init_args.batch_size', 'model.init_args.batch_size')
        
        if self.extra_args:
            for args, kwargs in self.extra_args:
                parser.add_argument(*args, **kwargs)
            
    def before_fit(self):
        self.trainer.logger.log_hyperparams(self.config)