import copy
import subprocess
import itertools
import warnings
import yaml
import argparse

def load_yaml(path):
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

def submit_experiments_main(
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
    jobs = list(itertools.product(*train_commands))

    if len(jobs) > max_job_limit:
        raise RuntimeError(
            f"""
            Number of jobs configured to run '{len(jobs)} exceeds 'max_job_limit' of {max_job_limit}.
                Look over the job configuration and if number of jobs is correct consider raising 'max_job_limit'.
            """)
    elif max_job_limit - 1 <= len(jobs) <= max_job_limit:
        warnings.warn(
            f"""
            Number of jobs is close to limit configured meaning the configuration file supplied
                may not be performing as expected, othewise you can ignore this message. 
                Number of jobs: {len(jobs)}
            """)

    command = ('sbatch', 'scripts/run-snakemake.sh')
    init_config = config
    for i, job in enumerate(jobs):
        config = copy.deepcopy(init_config)
        config['train_command'] = f"{' '.join([arg for arg, _ in job])}"
        config['run_name'] += '.' + '.'.join([name for _, name in job if name])
        config_args = list(f"{key}={value}" for key, value in config.items())

        print(f"Overriden config properties for experiment {i}:")
        for config_arg in config_args:
            print('\t', '\n\t\t'.join(config_arg.split('=')))

        commands = [*command, '--config', *config_args]
        if not preview:
            subprocess.run(commands)

def main():

    parser = argparse.ArgumentParser("Submit experiments")
    parser.add_argument("-c", "--config_file", type=str, default="experiments.yaml", help="Path to configuration file.")
    parser.add_argument("-m", "--max_job_limit", type=int, default=3, help="Max number of jobs capable of outputting without failure.")
    parser.add_argument("-p", "--preview", action='store_true', help="Do not run subprocces only preview job configurations.")
    args = parser.parse_args()

    submit_experiments_main(
        config_file=args.config_file,
        max_job_limit=args.max_job_limit,
        preview=args.preview,
    )

if __name__ == "__main__":
    main()




