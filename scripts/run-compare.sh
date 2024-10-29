#!/bin/bash

if [ -z "$1" ]; then
  echo "No argument provided for the name of the experiment!"
  exit 1
fi

if [ -z "$2" ]; then
  echo "No argument provided for model config filename!"
  exit 1
fi

for file in "$2"/*.yaml
do
    filename=$(basename "$file" .yaml)
    echo "Processing: $filename"
    sbatch scripts/run-snakemake.sh --config \
        experiment_name=$1\
        run_name=$filename \
        root_dir=lightning_logs \
        train_command=\
"\
fit \
--data configs/data/local.yaml \
--model $file \
--trainer.max_epochs 5 \
"

done
