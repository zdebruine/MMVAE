from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from jsonargparse import lazy_instance


def get_default_model_checkpoint():
    return lazy_instance(
        ModelCheckpoint,
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename="model_ckpt_{epoch}",
        every_n_epochs=1,
    )


def get_default_early_stopping():
    return lazy_instance(
        EarlyStopping,
        monitor="val_loss",
        min_delta=0.001,
        patience=2,
    )
