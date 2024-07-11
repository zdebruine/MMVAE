import subprocess
from typing import Any
import itertools
import os
import yaml

def submit_job(args):
    subprocess.run(["sbatch", "experiment/submit.sh"] + args)
    
def generate_combinations(config):
    combinations = []
    for basekey, val in config.items():
        key = f"{'-' if len(basekey) == 1 else '--'}{basekey}"
        if isinstance(val, list):
            combinations.append([(key, "", v) for v in val])
        elif isinstance(val, dict):
            if 'track' in val:
                for tk, tv in val['track'].items():
                    if isinstance(tv, list):
                        combinations.append([(key, f"{tk}_{v}", v) for v in tv])
                    else:
                        combinations.append([(key, tk, tv)])
            else:
                combinations.append([(key, "", val)])
        else:
            combinations.append([(key, "", val)])
            
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
        
        lightning_fit_args = " ".join([f"{key} {value}" for key, _, value in job])
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