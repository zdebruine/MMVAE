#!/bin/bash

python -m sciml.cli $@ \
    --trainer /Users/jaggerdenhof/Workspaces/repos/sci-ml/configs/defaults/trainer_cellx_config.yaml \
    --model /Users/jaggerdenhof/Workspaces/repos/sci-ml/configs/defaults/model_vae_config.yaml \
    --data /Users/jaggerdenhof/Workspaces/repos/sci-ml/configs/defaults/data_cellx_config.yaml \