import click
import os
import re


def find_last_job(rule_directory, return_all = False):
    # Regular expression to match job files and extract job numbers
    job_pattern = re.compile(r'job\.(\d+)\.(out|err)$')
    
    highest_jobnumber = -1
    last_job_out = None
    last_job_err = None

    # List all files in the rule directory
    for filename in os.listdir(rule_directory):
        match = job_pattern.match(filename)
        if match:
            jobnumber = int(match.group(1))
            if jobnumber > highest_jobnumber:
                highest_jobnumber = jobnumber
                last_job_out = os.path.join(rule_directory, f'job.{jobnumber}.out')
                last_job_err = os.path.join(rule_directory, f'job.{jobnumber}.err')

    if highest_jobnumber == -1:
        print("No job files found in the specified directory.")
        return None, None

    if return_all:
        return highest_jobnumber, last_job_out, last_job_err
    return highest_jobnumber

def display_job_output(stdout_file_path, stderr_file_path):
    try:
        # Read and display stdout
        with open(stdout_file_path, 'r') as stdout_file:
            stdout_content = stdout_file.read()
            print("=== STDOUT ===")
            print(stdout_content)
        
        # Read and display stderr
        with open(stderr_file_path, 'r') as stderr_file:
            stderr_content = stderr_file.read()
            print("\n=== STDERR ===")
            print(stderr_content)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

@click.command()
@click.option("-r", "--rule", type=str, default="submission", help="Rule to preview error logs.")
@click.option("--directory", type=click.Path(), default=".cmmvae/logs", show_default=True, help="Directory where logs are stored.")
def last(rule, directory):
    """Preview last config file"""
    rule_direcotry = os.path.join(directory, rule)
    jobid, out_file, err_file, = find_last_job(rule_direcotry, return_all=True)
    print(f"Previewing jobid {jobid}")
    display_job_output(out_file, err_file)

@click.group()
def logger():
    """Review logs from cmmvae output!"""
    
logger.add_command(last)

def main():
    logger()

if __name__ == '__main__':
    main()