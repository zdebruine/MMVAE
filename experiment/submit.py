import subprocess
from typing import Any
import itertools
import os
import yaml

def submit_job(args):
    subprocess.run(["sbatch", "experiment/submit.sh"] + args)
    
def parse(v):
    if isinstance(v, list):
        return '"' + '[' + ', '.join(f'"{str(elem)}"' if isinstance(elem, str) else str(elem) for elem in v) + ']' + '"'
    return v
    
def parse_value(argkey, value):
    if isinstance(value, dict):
        return ' '.join(f"{argkey}.{k} {parse(v)}" for k, v in value.items())
    elif isinstance(value, list):
        return ' '.join(v for v in value)
    return f"{argkey} {value}"
        
def get_tracked_command_line_options(argkey, config):
    if len(config) == 1:
        key, value = next(iter(config.items()))
        if not isinstance(value, list):
            return [(argkey, key, value)]
        return [(argkey, f"{key}_{v}" if v and not isinstance(v, bool) else key, v) for v in value]
    else:
        return [(argkey, key, value) for key, value in config.items()]
        
def get_lightning_fit_args_combos(config: dict[str, Any]):
    
    combinations = []
    for key, value in config.items():
        argkey = f"-{key}" if len(key) == 1 else f"--{key}"
        
        if value == 'flag':
            combinations.append([(argkey, "", "")])
            continue
        
        if isinstance(value, dict) and 'track' in value:
            combinations.append(get_tracked_command_line_options(argkey, value['track']))
        else:
            combinations.append([(argkey, "", value)])
            
    combinations = list(itertools.product(*combinations))
    return combinations

def get_snakemake_config_args(config: dict[str, Any]):
    args = []
    for key, value in config.items():
        if key in ('lightning_fit_args', 'run_name'):
            continue
        args.append(f"{key}={value}")
    return args

def get_snakemake_args(config):
    args = []
    for key, value in config.items():
        argkey = f"-{key}" if len(key) == 1 else f"--{key}"
        if value == 'flag':
            args.append(argkey)
    return args

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
            
    # submit_main(config)
    jobs = get_lightning_fit_args_combos(config['lightning_fit_args'])
    snakemake_config_args = get_snakemake_config_args(config)
    snakemake_args = get_snakemake_args(config)
    default_run_name = config.get('run_name', '')
    
    for job in jobs:
        lightning_fit_args = ' '.join(parse_value(argkey, value) for argkey, _, value in job)
        run_name = '_'.join(name for _, name, _ in job if name)
        args = ['--config', f"lightning_fit_args={lightning_fit_args}", f"run_name={default_run_name + '_' if default_run_name else ''}{run_name}", *snakemake_config_args, *snakemake_args, ]
        print(args)
        submit_job(args)