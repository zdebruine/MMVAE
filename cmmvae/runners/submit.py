import click
import os
from ._decorators import click_env_option

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

def submit_experiments(
    config_file: str,
    max_job_limit: int,
    preview: bool = False,
):
    config: dict = load_yaml(config_file)

    train_commands = []
    subcommand, commands = next(iter(config['train_command'].items()))
    for command_key, command in commands.items():
        if isinstance(command, dict):
            if 'track' in command:
                command = command['track']
                if isinstance(command, dict) and len(command) == 1 and isinstance(list(command.values())[0], list):
                    train_commands.append([(f"--{command_key} {val}", name.replace("{value}", str(val))) for name, val in command.items()])
                elif isinstance(command, dict):
                    train_commands.append([(f"--{command_key} {parse(val)}", name) for name, val in command.items()])
                else:
                    train_commands.append([(f"--{command_key} {command}")])
            else:
                raise RuntimeError()
        else:
            train_commands.append([(f"--{command_key} {parse(command)}", "")])
    from itertools import product
    jobs = list(product(*train_commands))

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

    command = ('sbatch', 'scripts/run-snakemake.sh')
    init_config = config
    from copy import deepcopy
    for i, job in enumerate(jobs):
        config = deepcopy(init_config)
        config['train_command'] = f"{subcommand} {' '.join([arg for arg, _ in job])}"
        config['run_name'] += '.' + '.'.join([name for _, name in job if name])
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
@click_env_option("-c", "--config_file", type=str, default="experiments.yaml", help="Path to configuration file.")
@click_env_option("-m", "--max_job_limit", type=int, default=3, help="Max number of jobs capable of outputting without failure.")
@click_env_option("-p", "--preview", is_flag=True, help="Do not run subprocess, only preview job configurations.")
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
