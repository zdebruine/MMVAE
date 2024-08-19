import sys
import click
import os
from datetime import datetime
from copy import deepcopy
import subprocess

def load_yaml(path):
    import yaml
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def combine_keys(*keys, sep: str = "."):
    keys = [key for key in keys if key]
    return sep.join(keys)

def parse(v):
    if isinstance(v, list):
        return '"' + '[' + ', '.join(f'"{str(elem)}"' if isinstance(elem, str) else str(elem) for elem in v) + ']' + '"'
    return v

def parse_tracked_command(command_key, command):
    commands = []
    if isinstance(command, dict) and len(command) == 1 and isinstance(list(command.values())[0], list):
        for name, val in command.items():
            commands.append((command_key, val, name.replace("{value}", str(val))))
    elif isinstance(command, dict):
        for name, val in command.items():
            commands.append((command_key, parse(val), name))
    else:
        raise RuntimeError(
            """
            Unintened code block hit during execution.
            Please file a issue to the maintainers if 
            applicable.
            """
        )
    return commands

def key_to_command(command_key: str):
    if command_key.startswith('_'):
        key = '---'
    else:
        key = '-' if len(command_key) == 1 else "---"
    return f"{key}{command_key}"
            
class Experimenter:
    
    def __init__(
        self,
        config_file: str,
        config: dict,
        max_job_limit: int,
        timestamp: bool = False,
        preview: bool = False,
        command: str = ("sbatch", "scripts/run-snakemake.sh"),
    ): 

        self.config: dict = load_yaml(config_file)
        self.config.update(config)
        self.max_job_limit = max_job_limit
        self.timestamp = timestamp
        self.ispreview = preview
        self.command = command
            
        self.setup_experiments()
        self.validate_experiments()
        self.build_job_commands()
        
        if self.ispreview:
            for i, command in enumerate(self.job_commands, start=1):
                print(f"Experiment {i} commands:\n\t{' '.join(command)}")
        
        
    def build_job_commands(self):
        self.job_commands = []
        for job in self.jobs:
            run_config = deepcopy(self.config)
            train_command_string = ' '.join([f"{'--' if len(command) > 1 else '-'}{command} {value}" for command, value, _ in job])
            run_config['train_command'] = f"{self.subcommand} {train_command_string}"
            run_name = '.' + '.'.join([name for _, _, name in job if name])
            if self.timestamp:
                run_name += datetime.now().strftime("%Y%m%d_%H%M%S")
            run_config['run_name'] += run_name
            
            config_args = list(f"{key}={value}" for key, value in run_config.items())

            self.job_commands.append([*self.command, '--config', *config_args])
        
    def setup_experiments(self):
        
        self.subcommand, train_commands = next(iter(self.config['train_command'].items()))
        command_pairs = self.parse_command_key_value_name_combinations(train_commands)
        from itertools import product
        self.jobs = list(product(*command_pairs))
        
    def parse_command_key_value_name_combinations(self, commands: dict):
        combinations = []
        for command_key, command in commands.items():
            if isinstance(command, dict):
                if 'track' in command:
                    command = command['track']
                    tracked_commands = parse_tracked_command(command_key, command)
                    combinations.append(tracked_commands)
                else:
                    raise RuntimeError()
            else:
                combinations.append([(command_key, parse(command), "")])
        return combinations
    
    def validate_experiments(self):
        
        if not hasattr(self, 'jobs'):
            raise RuntimeError("setup_experiments failed: attr 'jobs' not available")
        
        if len(self.jobs) > self.max_job_limit:
            raise RuntimeError(
                f"""
                Number of jobs configured to run '{len(self.jobs)} exceeds 'max_job_limit' of {self.max_job_limit}.
                    Look over the job configuration and if number of jobs is correct consider raising 'max_job_limit'.
                """)
        elif self.max_job_limit - 1 <= len(self.jobs):
            import warnings
            warnings.warn(
                f"""
                Number of jobs is close to limit configured meaning the configuration file supplied
                    may not be performing as expected, otherwise you can ignore this message. 
                    Number of jobs: {len(self.jobs)}
                """)
            
    def run(self):
        for i, commands in enumerate(self.job_commands):
            print(f"Job: {i}", '\n\t'.join(commands))
            if not self.ispreview:
                subprocess.run(commands)
        print(f"Total jobs ran: {len(self.job_commands)}")
        
def parse_kwargs(ctx, param, value):
    kwargs = {}
    for item in value:
        try:
            key, val = item.split('=', 1)
            kwargs[key] = val
        except ValueError:
            raise click.BadParameter(f'Invalid format for {param.name}. Expected format: key=value')
    return kwargs


@click.command()
@click.option("--config_file", type=str, default="experiments.yaml", show_default=True, help="Path to configuration file.")
@click.option('--config', multiple=True, callback=parse_kwargs, help="Configuration options as key=value pairs")
@click.option("-m", "--max_job_limit", type=int, default=3, show_default=True, help="Max number of jobs capable of outputting without failure.")
@click.option("-t", "--timestamp", is_flag=True, help="Added timestamp to end of run name.")
@click.option("-p", "--preview", is_flag=True, help="Do not run subprocess, only preview job configurations.")
def submit(**kwargs):
    """
    Submit experiments using configurations from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file.
        max_job_limit (int): Maximum number of jobs that can be run.
        preview (bool): Whether to preview job configurations without running them.
    """
    Experimenter(**kwargs).run()

@click.group()
def experiment():
    """Submit snakemake experiments"""

experiment.add_command(submit)

def main():
    experiment()
    
if __name__ == "__main__":
    main()
