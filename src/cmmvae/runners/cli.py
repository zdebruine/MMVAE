"""
    Subclass of LightningCli specialized for pipeline.
"""
import os
import sys
import shutil

import click

from lightning.pytorch import cli as plcli
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from cmmvae.callbacks import (
    PredictionWriter,
)


class CMMVAECli(plcli.LightningCLI):
    """
    LightningCLI meant to ease in setting default arguments and
    logging parameters. All Models of subclass BaseVAEModel should use this
    """

    def __init__(self, moniter="val_loss", only_data=False, **kwargs):
        """
        Handles loading trainer, model, and data modules from config file,
        while linking common arguments for ease of access.
        """
        self.moniter = moniter
        self.running_subcommands = bool(kwargs.get("run", True)) or only_data
        self.only_data = only_data

        if "parser_kwargs" not in kwargs:
            kwargs["parser_kwargs"] = {}

        kwargs["parser_kwargs"].update(
            {
                "default_env": True,
                "parser_mode": "omegaconf",
            }
        )

        if "trainer_defaults" not in kwargs:
            kwargs["trainer_defaults"] = {
                "logger": {"class_path": "lightning.pytorch.loggers.TensorBoardLogger"},
                "enable_progress_bar": False,
            }

        super().__init__(**kwargs)

    def before_instantiate_classes(self) -> None:
        if self.subcommand == "predict":
            self.save_config_callback = None
        if self.only_data:
            if "model" in self.config:
                del self.config["model"]
            if "trainer" in self.config:
                del self.config["trainer"]

    def add_arguments_to_parser(self, parser):
        # Add arguments for logging and ease of access
        parser.add_argument(
            "--default_root_dir", required=True, help="Default root directory"
        )
        parser.add_argument(
            "--experiment_name", required=True, help="Name of experiment directory"
        )
        parser.add_argument(
            "--run_name", required=True, help="Name of the experiment run"
        )

        # Link ease of access arguments to their respective targets
        parser.link_arguments("default_root_dir", "trainer.default_root_dir")
        parser.link_arguments("default_root_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("experiment_name", "trainer.logger.init_args.name")
        parser.link_arguments("run_name", "trainer.logger.init_args.version")

        # Allow for ckpt_path if run=False
        if not self.running_subcommands:
            parser.add_argument("--ckpt_path", help="ckpt path to be passed to model")

        # Add ModelCheckpoint callback to trainer
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "model_checkpoint.monitor": self.moniter,
                "model_checkpoint.filename": "model_chkpt_{epoch}",
                "model_checkpoint.save_top_k": 3,
                "model_checkpoint.save_last": True,
                "model_checkpoint.mode": "min",
            }
        )

        # Add EarlyStopping callback to trainer
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": self.moniter,
                "early_stopping.mode": "min",
                "early_stopping.patience": 2,
                "early_stopping.min_delta": 0.001,
                "early_stopping.strict": False,
            }
        )

        # Add PredictionWriter callback to trainer
        parser.add_lightning_class_args(PredictionWriter, "prediction_writer")
        parser.link_arguments("default_root_dir", "prediction_writer.root_dir")
        parser.link_arguments("experiment_name", "prediction_writer.experiment_name")
        parser.link_arguments("run_name", "prediction_writer.run_name")
        parser.link_arguments(
            "data.init_args.conditionals_directory",
            "model.init_args.module.init_args.vae.init_args.conditionals_directory",
        )

    def before_fit(self):
        """Save model parameters and prints model configuration before fit"""
        # print(self.model)
        self.trainer.logger.log_hyperparams(self.config)

    def after_fit(self):
        """Runs predict subcommand after fit with best ckpt_path."""

        if not self.trainer.checkpoint_callback:
            import warnings

            warnings.warn(
                "Skipping predict step after fit: checkpoint_callback not defined"
            )
            return

        best_model_path = self.trainer.checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(
            os.path.dirname(best_model_path), "best_model.ckpt"
        )

        shutil.copy(src=best_model_path, dst=new_best_model_path)

        self.config["fit"]["default_root_dir"]

        self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=best_model_path,
        )


def context_settings(**kwargs):
    settings = dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
    settings.update(kwargs)
    return settings


@click.command(context_settings=context_settings())
@click.pass_context
def cli(ctx: click.Context, **kwargs):
    sys.argv = [sys.argv[0]]
    return CMMVAECli(args=ctx.args, **kwargs)


if __name__ == "__main__":
    cli()
