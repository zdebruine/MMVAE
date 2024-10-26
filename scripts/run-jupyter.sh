#!/bin/bash
#SBATCH --job-name=jupyter_notebook      # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=128G                        # Total memory
#SBATCH --time=02:00:00                  # Time limit hrs:min:sec
#SBATCH --output=.cmmvae/logs/jupyter_notebook/job.%j.out
#SBATCH --error=.cmmvae/logs/jupyter_notebook/job.%j.err

scripts/run-command-n-env.sh jupyter notebook --no-browser --ip=0.0.0.0 "$@"
