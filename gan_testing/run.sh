#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --mem=6144 ##Memory I want to use in MB

#SBATCH --time=02:00:00 ## time it will take to complete job

#SBATCH --partition=gpu ##Partition I want to use

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=gan_test ## Name of job

#SBATCH --output=discrim.%j.out ##Name of output file

module load ml-python/nightly
source /active/debruinz_project/jack_lukomski/jacks_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/active/debruinz_project/jack_lukomski/MMVAE_Adversarial_Team
pip install torchdata
pip install scipy
pip install -U scikit-learn
/active/debruinz_project/jack_lukomski/jacks_venv/bin/python gan_main.py
