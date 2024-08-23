#!/bin/bash
#SBATCH --job-name=submission
#SBATCH --output=.cmmvae/logs/submission/job.%j.out
#SBATCH --error=.cmmvae/logs/submission/job.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=bigmem
#SBATCH --mem=4G

if [ -z "${CMMVAE_ENV_PATH}" ]; then
    echo "CMMVAE_ENV_PATH is not set. Please set CMMVAE_ENV_PATH to virtual enviroment path."
    exit 1
fi

module purge

source $CMMVAE_ENV_PATH/bin/activate
snakemake --profile workflow/profile/slurm --latency-wait 30 "$@" 
deactivate