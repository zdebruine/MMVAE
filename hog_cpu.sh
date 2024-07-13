#!/bin/bash
#SBATCH --job-name=_submission
#SBATCH --output=.snakemake/slurm_logs/job.%j.out
#SBATCH --error=.snakemake/slurm_logs/job.%j.err
#SBATCH --time=1-00:00:00  # Set a time limit for the job
#SBATCH --mem=1G         # Memory allocation
#SBATCH --cpus-per-task=1 # CPU allocation
#SBATCH --partition=cpu

while true
do
    sleep 1
done