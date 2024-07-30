#!/bin/bash
#SBATCH --job-name=submission
#SBATCH --output=.snakemake/slurm_logs/submission/job.%j.out
#SBATCH --error=.snakemake/slurm_logs/submission/job.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=bigmem
#SBATCH --mem=4G


snakemake --profile workflow/profile/slurm --keep-incomplete --latency-wait 30 "$@" 
