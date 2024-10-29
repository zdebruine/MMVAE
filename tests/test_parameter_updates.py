# Since we can't import your modules, we'll define minimal versions for testing
from cmmvae.runners.cli import CMMVAECli
from cmmvae.models import CMMVAEModel

# Minimal FCBlockConfig and FCBlock for testing


def __test_parameter_updates():
    # Create dummy data
    cli = CMMVAECli(run=None, save_config_callback=None)

    model: CMMVAEModel = cli.model
    trainer = cli.trainer
    datamodule = cli.datamodule

    datamodule.species = [
        species for species in datamodule.species if species.name == "human"
    ]

    trainer.limit_train_batches = 1
    trainer.limit_val_batches = 0
    trainer.fit_loop.max_epochs = 1
    logger = trainer.logger
    trainer._logger_connector.on_trainer_init(logger, 1)

    for callback in trainer.early_stopping_callbacks:
        callback.strict = False

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    __test_parameter_updates()
