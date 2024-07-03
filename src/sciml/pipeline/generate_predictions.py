import os
import sys
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Optional
from sciml.cli import SCIMLCli
from .generate_umap import plot_umap


def generate_predictions(
    ckpt_path: str, 
    config_path: str,
    z_embeddings_name: str = 'z_embeddings.npz',
    metadata_name: str = 'metadata.pkl',
    result_directory: str = None,
):
    
    for path in (ckpt_path, config_path):
        if not os.path.exists(path):
            raise FileExistsError(path)

    config = None
    with open(config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file)
        
    # Doesn't expect ckpt_path when run=False
    if 'ckpt_path' in config:
        del config['ckpt_path']
        
    if 'trainer' in config and 'logger' in config['trainer'] and 'init_args' in config['trainer']['logger']:
        config['trainer']['logger']['init_args']['log_graph'] = False
        
    cli = SCIMLCli(run=False, args=config, save_config_callback=None) # don't save config for predictions run
    if hasattr(cli.model, 'save_predictions'):
        cli.model.save_predictions = False # prevent model from saving predictions to disk
    predictions = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule, return_predictions=True, ckpt_path=ckpt_path)
    
    z_embeddings, metadata = cli.model.parse_predictions(predictions)
    
    if result_directory == None:
        result_directory = cli.trainer.logger.log_dir
        
    z_embeddings_path = os.path.join(result_directory, z_embeddings_name)
    metadata_path = os.path.join(result_directory, metadata_name)
    
    np.savez(z_embeddings_path, z_embeddings)
    metadata.to_pickle(metadata_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True, help="Directory path for the experiment")
    parser.add_argument('-c', '--ckpt_name', type=str, default='last.ckpt', help="Path to the checkpoint to get predictions from (Default: last)")
    parser.add_argument('--pipeline_directory', type=str, help='Directory for the pipeline results')
    args = parser.parse_args()
    # remove all args passed in for LightningCli
    sys.argv = [sys.argv[0]]
    
    ckpt_path = os.path.join(args.directory, 'checkpoints', args.ckpt_name)
    config_path = os.path.join(args.directory, 'config.yaml')
    print(config_path)
    pipeline_directory = args.pipeline_directory if args.pipeline_directory else args.directory
    
    generate_predictions(ckpt_path, config_path, result_directory=pipeline_directory)
    
        
        
        
        
        
        

    
