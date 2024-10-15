#!/bin/bash
#SBATCH --job-name=snakemake
#SBATCH --output=.cmmvae/logs/snakemake/job.%j.out
#SBATCH --error=.cmmvae/logs/snakemake/job.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=bigmem
#SBATCH --mem=1G

scripts/run-command-n-env.sh snakemake --profile workflow/profile/slurm --latency-wait 60 "$@"
