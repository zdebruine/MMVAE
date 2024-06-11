#!/bin/bash

if ! [ "$#" -eq 2 ]; then
    echo "Please provide subcommand and yaml name with out .yaml"
    exit 1
fi

python -m sciml.main $1 --print_config \
    --trainer $SCIML_REPO_PATH/configs/defaults/trainer_cellx_config.yaml \
    --model $SCIML_REPO_PATH/configs/defaults/model_vae_config.yaml \
    --data $SCIML_REPO_PATH/configs/defaults/data_cellx_config.yaml \
    > $SCIML_REPO_PATH/configs/$2.yaml

echo "Config file: $SCIML_REPO_PATH/configs/$2.yaml"