#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --mem=6144 ##Memory I want to use in MB

#SBATCH --time=00:10:00 ## time it will take to complete job

#SBATCH --partition=gpu ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=jh ## Name of job

#SBATCH --output=test-job.%j.out ##Name of output file

module load ml-python/nightly
export PYTHONPATH=/home/howlanjo/dev/MMVAE:$PYTHONPATH
python /home/howlanjo/dev/MMVAE/get-started/howie.py