#!/bin/bash

python -m sciml.cli $@ \
    --trainer $SCIML_REPO_PATH/configs/defaults/trainer_cellx_config.yaml \
    --model $SCIML_REPO_PATH/configs/defaults/model_vae_config.yaml \
    --data $SCIML_REPO_PATH/configs/defaults/data_cellx_config.yaml \