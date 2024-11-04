#!/bin/bash

if [ -z "$1" ]; then
  echo "No argument provided for the name of the experiment!"
  exit 1
fi

if [ -z "$2" ]; then
  echo "No argument provided for model config filename!"
  exit 1
fi

experiment_name=$1
compare_dir=$2

shift 2

root_dir=lightning_logs
data=configs/data/local.yaml
max_epochs="--trainer.max_epochs 5"

for file in $compare_dir/*.yaml
do
  run_name=$(basename "$file" .yaml)
  version=$(ls -d "$root_dir/$experiment_name/$base_run_name"* | grep -E 'V[0-9]{3}$' | sort -V | tail -n 1 | sed -E 's/V([0-9]{3})$/\1/' | awk '{printf "V%03d", $1 + 1}')

  echo "Processing: $file"
  sbatch scripts/run-snakemake.sh --config \
    root_dir="$root_dir" \
    experiment_name="$experiment_name" \
    run_name="${run_name}.${version}" \
    train_command="fit --data $data --model $file $max_epochs $*"
done
