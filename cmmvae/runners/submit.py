import click
import os
from datetime import datetime

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
        
def create_experiment_combinations(combinations):
    from itertools import product
    return list(product(*combinations))

def parse_train_commands(commands):
    combinations = []
    for command_key, command in commands.items():
        if isinstance(command, dict):
            if 'track' in command:
                command = command['track']
                tracked_commands = parse_tracked_command(command)
                combinations.append(tracked_commands)
            else:
                raise RuntimeError()
        else:
            combinations.append([(command_key, parse(command), "")])
    return create_experiment_combinations(combinations)

def validate_experiments(jobs, max_job_limit):
    if len(jobs) > max_job_limit:
        raise RuntimeError(
            f"""
            Number of jobs configured to run '{len(jobs)} exceeds 'max_job_limit' of {max_job_limit}.
                Look over the job configuration and if number of jobs is correct consider raising 'max_job_limit'.
            """)
    elif max_job_limit - 1 <= len(jobs):
        import warnings
        warnings.warn(
            f"""
            Number of jobs is close to limit configured meaning the configuration file supplied
                may not be performing as expected, otherwise you can ignore this message. 
                Number of jobs: {len(jobs)}
            """)
        
def submit_experiments(
    config_file: str,
    max_job_limit: int,
    preview: bool = False,
    timestamp: bool = False,
):
    config: dict = load_yaml(config_file)
    
    subcommand, train_commands = next(iter(config['train_command'].items()))
    subcommand, jobs = parse_train_commands(train_commands)
    
    validate_experiments(jobs, max_job_limit)
    
    
    command = ('sbatch', 'scripts/run-snakemake.sh')
    init_config = config
    from copy import deepcopy
    for i, job in enumerate(jobs):
        
        config = deepcopy(init_config)
        train_commands = [f"{'--' if len(command) > 1 else '-'}{command} {value}" for command, value, _ in job]
        config['train_command'] = f"{subcommand} {' '.join(train_commands)}"
        
        run_name = '.' + '.'.join([name for _, _, name in job if name])
        if timestamp:
            run_name += datetime.now().strftime("%Y%m%d_%H%M%S")
        config['run_name'] += run_name
        
        config_args = list(f"{key}={value}" for key, value in config.items())

        print(f"Overriden config properties for experiment {i}:")
        for config_arg in config_args:
            print('\t', '\n\t\t'.join(config_arg.split('=')))

        commands = [*command, '--config', *config_args]
        if not preview:
            import subprocess
            subprocess.run(commands)
        else:
            print("Total jobs found:", len(jobs))


@click.command()
@click.option("-c", "--config_file", type=str, default="experiments.yaml", help="Path to configuration file.")
@click.option("-m", "--max_job_limit", type=int, default=3, help="Max number of jobs capable of outputting without failure.")
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
    submit_experiments(**kwargs)

def main():
    submit()
    
if __name__ == "__main__":
    main()
