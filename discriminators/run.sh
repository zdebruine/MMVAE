#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --mem=1024 ##Memory I want to use in MB

#SBATCH --time=00:05:00 ## time it will take to complete job

#SBATCH --partition=all ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=mmvae_adversarial_team ## Name of job

#SBATCH --output=mmvae_adversarial_team.%j.out ##Name of output file

module load ml-python/nightly
export PYTHONPATH=$PYTHONPATH:~/MMVAE_Adversarial_Team
python discriminator.py
