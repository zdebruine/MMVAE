from .cli import SCIMLCli



if __name__ == "__main__":
    
    cli = SCIMLCli()
    
    if cli.model.save_predictions:
        cli.trainer.predict()