#!/bin/bash

debug=false
append_commit_hash=true
root_dir="${CMMVAE_ROOT_DIR}"
experiment="${CMMVAE_EXPERIMENT_NAME}"
data="${CMMVAE_DATA_CONFIG}"
compare=""
max_epochs="${CMMVAE_MAX_EPOCHS}"
commit_hash=""
extra_args=""

if [ -z "${max_epochs}" ]; then
  max_epochs=5
else
  echo "CMMVAE_MAX_EPOCHS is set to '$CMMVAE_MAX_EPOCHS'"
fi

if [ -z "${data}" ]; then
  data=configs/data/local.yaml
else
  echo "CMMVAE_DATA_CONFIG is set to '$CMMVAE_DATA_CONFIG'"
fi

if [ -z "${root_dir}" ]; then
  root_dir=lightning_logs
else
  echo "CMMVAE_ROOT_DIR is set to '$CMMVAE_ROOT_DIR'"
fi

if [ -z "${experiment}" ]; then
  experiment=default
else
  echo "CMMVAE_EXPERIMENT_NAME is set to '$CMMVAE_EXPERIMENT_NAME'"
fi

for arg in "$@"
do
    case $arg in
        --debug)
            debug=true
            shift
            ;;
        --no-commit-hash)
            append_commit_hash=false
            shift
            ;;
        root_dir=*)
            root_dir="${arg#*=}"
            shift
            ;;
        experiment=*)
            experiment="${arg#*=}"
            shift
            ;;
        compare=*)
            compare="${arg#*=}"
            shift
            ;;
        data=*)
            data="${arg#*=}"
            shift
            ;;
        max_epochs=*)
            max_epochs="${arg#*=}"
            shift
            ;;
        *)
            extra_args="$extra_args $arg"
            shift
            ;;
    esac
done

if [ -z "$compare" ]; then
  echo "Please specify directory that contains model configs to compare."
  exit 1
fi

if [ "$append_commit_hash" = true ]; then
  if ! command -v git &> /dev/null; then
      echo "Error: Git is not installed. Please install Git to use this script or specify --no-commit-hash."
      exit 1
  fi

  if ! git rev-parse --is-inside-work-tree &> /dev/null; then
      echo "Error: This is not a Git repository. Please run the script inside a Git repository or specify --no-commit-hash."
      exit 1
  fi

  commit_hash=$(git rev-parse --short HEAD)
  echo "Latest Commit Hash: $commit_hash"
else
  echo "Skipping commit hash display."
fi

for file in "$compare"/*.yaml
do
  run_name=$(basename "$file" .yaml)

  if [ "$commit_hash" != "" ]; then
    run_name="${commit_hash}.${run_name}"
  fi

  ran_dirs=$(ls -d "$root_dir/$experiment/$run_name"* 2>/dev/null)

  if [ -z "$ran_dirs" ]; then
    version="V000"
  else
    version=$(echo "$ran_dirs" | grep -E 'V[0-9]{3}$' | sort -V | tail -n 1 | sed -E 's/.*V([0-9]{3})$/\1/' | awk '{printf "V%03d", $1 + 1}')
  fi

  echo "Processing: $file"
  command="scripts/run-snakemake.sh --config \
    root_dir=${root_dir} \
    experiment_name=${experiment} \
    run_name=${version}.${run_name} \
    train_command=\"fit --model $file --data $data --trainer.max_epochs $max_epochs $extra_args\"
  "
  echo $command
  if [ "$debug" = false ]; then
    eval "sbatch $command"
  fi
done
