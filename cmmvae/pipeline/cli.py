
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from cmmvae.models import BaseModel


class SCIMLCli(LightningCLI):
    """
        LightningCLI meant to ease in setting default argumetns and 
    logging parameters. All Models of subclass BaseVAEModel should use this
    cli.
    """
    def __init__(
        self, 
        extra_parser_kwargs = {},
        **kwargs
    ):
        self.is_run = bool(kwargs.get('run', True)) 
        
        super().__init__(
            model_class=BaseModel,
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
        parser.add_argument('--predict_dir', required=True, help="Where to store predictions after fit")
        
        if self.is_run == False:
            parser.add_argument('--ckpt_path', required=True, help="Ckpt path to be passed to model")
        
        # Link ease of access arguments to their respective targets
        parser.link_arguments('default_root_dir', 'trainer.default_root_dir')
        parser.link_arguments('default_root_dir', 'trainer.logger.init_args.save_dir')
        parser.link_arguments('experiment_name', 'trainer.logger.init_args.name')
        parser.link_arguments('run_name', 'trainer.logger.init_args.version')
        parser.link_arguments('predict_dir', 'model.init_args.module.init_args.predict_dir')
        parser.link_arguments('data.init_args.batch_size', 'model.init_args.batch_size')
        
    def before_fit(self):
        print(self.model)
        self.trainer.logger.log_hyperparams(self.config)
        
    def after_fit(self):
        self.model.predict_dir = self.config['fit']['predict_dir']
        self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path='best',
        )

if __name__ == "__main__":
    
    SCIMLCli()