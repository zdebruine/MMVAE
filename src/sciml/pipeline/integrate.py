from typing import Any
from sciml.cli import SCIMLCli
import lightning as L
from sciml.models import BaseVAEModel



def integrate(
    trainer: L.Trainer, 
    model: BaseVAEModel, 
    datamodule: L.LightningDataModule,
    ckpt_path: str = None,
    predict_kwargs: dict[str, Any] = {}
):

    predictions = trainer.predict(
        model=model, 
        datamodule=datamodule, 
        return_predictions=True, 
        ckpt_path=ckpt_path)
    
    try:
        model.save_predictions(predictions, **predict_kwargs)
    except NotImplementedError as e:
        print("Predictions could not be saved from model save_predictions not implemented")
        raise e
    
if __name__ == "__main__":
    
    # don't save config for predictions run
    cli = SCIMLCli(run=False, save_config_callback=None, extra_arguments=[
        (('--save_prediction_kwargs',), {
            'default': {}, 
            'help': "Dictionary of kwargs to be passed to model save_predictions"
        })
    ])

    integrate(
        trainer=cli.trainer,
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=cli.config['ckpt_path'],
        predict_kwargs=dict(cli.config['save_prediction_kwargs'])
    )
        
        
        
        
        
        

    
