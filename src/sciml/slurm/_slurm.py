import os
import subprocess


from . import utils

def is_available():
    return utils.is_command_available('sbatch')

def submit_sbatch_job(
    script_path,
    job_name=None, output=None, error=None, time=None, partition=None,
    ntasks=None, cpus_per_task=None, mem=None, mail_user=None, mail_type=None,
    dependency=None, constraint=None, exclude=None, nodelist=None, nodes=None,
    ntasks_per_node=None, gpus=None, gres=None, account=None, qos=None,
    chdir=None, workdir=None, export=None, no_requeue=False, requeue=False,
    signal=None, test_only=False, licenses=None, ntasks_per_core=None, threads_per_core=None,
    distribution=None, oversubscribe=False, mem_per_cpu=None, mem_per_gpu=None,
    cpus_per_gpu=None, hint=None, contiguous=False, cpu_freq=None, exclusive=False,
    no_kill=False, no_multithread=False, overcommit=False, cores_per_socket=None,
    switches=None, bb=None, mcs_label=None, begin=None, deadline=None, hold=None,
    reboot=False, reservation=None, wait=False, wrap=None, parsable=False,
    profile=None, propagate=None, quiet=False, usage=False
):
    args = ['sbatch']
    
    arg_map = {
        'job_name': '--job-name', 'output': '--output', 'error': '--error',
        'time': '--time', 'partition': '--partition', 'ntasks': '--ntasks',
        'cpus_per_task': '--cpus-per-task', 'mem': '--mem', 'mail_user': '--mail-user',
        'mail_type': '--mail-type', 'dependency': '--dependency', 'constraint': '--constraint',
        'exclude': '--exclude', 'nodelist': '--nodelist', 'nodes': '--nodes',
        'ntasks_per_node': '--ntasks-per-node', 'gpus': '--gpus', 'gres': '--gres',
        'account': '--account', 'qos': '--qos', 'chdir': '--chdir', 'workdir': '--workdir',
        'export': '--export', 'no_requeue': '--no-requeue', 'requeue': '--requeue',
        'signal': '--signal', 'test_only': '--test-only', 'licenses': '--licenses',
        'ntasks_per_core': '--ntasks-per-core', 'threads_per_core': '--threads-per-core',
        'distribution': '--distribution', 'oversubscribe': '--oversubscribe',
        'mem_per_cpu': '--mem-per-cpu', 'mem_per_gpu': '--mem-per-gpu',
        'cpus_per_gpu': '--cpus-per-gpu', 'hint': '--hint', 'contiguous': '--contiguous',
        'cpu_freq': '--cpu-freq', 'exclusive': '--exclusive', 'no_kill': '--no-kill',
        'no_multithread': '--no-multithread', 'overcommit': '--overcommit',
        'cores_per_socket': '--cores-per-socket', 'switches': '--switches', 'bb': '--bb',
        'mcs_label': '--mcs-label', 'begin': '--begin', 'deadline': '--deadline',
        'hold': '--hold', 'reboot': '--reboot', 'reservation': '--reservation',
        'wait': '--wait', 'wrap': '--wrap', 'parsable': '--parsable', 'profile': '--profile',
        'propagate': '--propagate', 'quiet': '--quiet', 'usage': '--usage'
    }
    
    for key, value in locals().items():
        if key in arg_map and value is not None:
            args.append(arg_map[key])
            if not isinstance(value, bool):  # Boolean options are flags
                args.append(str(value))
        elif key in arg_map and value is True:
            args.append(arg_map[key])
    
    args.append(script_path)

    result = subprocess.run(args, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Job submitted successfully. Output: {result.stdout.strip()}")
    else:
        print(f"Job submission failed. Error: {result.stderr.strip()}")