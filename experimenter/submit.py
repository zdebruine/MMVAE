import subprocess
from typing import Any, Union
import itertools
import os
import yaml
from collections import OrderedDict

def submit_job(args):
    subprocess.run(["sbatch", "slurm/submit.sh"] + args)
    
def generate_combinations(config):
    combinations = []
    for key, val in config.items():
        track = False
        if key.endswith('.$track'):
            key = key.replace('.$track', '')
            track = True
        if len(key) == 1:
            arg_key = '-' + key
        else:
            arg_key = '--' + key
        if isinstance(val, list):
            combinations.append([(arg_key, "", v) for v in val])
        elif isinstance(val, dict):
            combinations.append([(arg_key, k if track else "", v) for k, v in val.items()])
        else:
            combinations.append([(arg_key, "", val)])
    combinations = list(itertools.product(*combinations))
    
    return combinations
    
def submit_main(config: dict[str, Any]):
    """Submit combinations of experiments to sbatch. Combinations are created by specifiying
    kwargs that are supported as a value, list, or dict. Values will be the same for every run.
    Lists will be turned into combinations of each value and dictionarys will be the same for their values
    the only difference is the keys for the dictionary will specifify the nameing value for the run name.
    (ie. config_path is long so can be shortened with dictionary keys)
    """
    
    jobs = generate_combinations(config['lightning_fit_args'])
    snakemake_args = [f"{key}={value}" for key, value in config.items() if not key in ('lightning_fit_args',)]
    for job in jobs:
        
        lightning_fit_args = " ".join([f"{argkey} {value}" for argkey, _, value in job])
        run_name = "_".join([name for _, name, _, in job if name])
        
        
        args = ['--config', f"lightning_fit_args=\"{lightning_fit_args}\"", f"run_name={run_name}", *snakemake_args]

        submit_job(args)

if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default="", help="path to mutator yaml file")
    args = parser.parse_args()
    
    if not (args.config_path and os.path.exists(args.config_path)):
        raise FileNotFoundError(args.config_path)
    
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if not config:
        raise RuntimeError("Config invalid yaml file or empty!")
            
    submit_main(config)