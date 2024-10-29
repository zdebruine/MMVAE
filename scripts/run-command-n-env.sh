#!/bin/bash

if [ -z "${CMMVAE_ENV_PATH}" ]; then
    echo "CMMVAE_ENV_PATH is not set. Please set CMMVAE_ENV_PATH to virtual enviroment path."
    exit 1
fi

if hash module 2>/dev/null; then
    module purge
fi

source $CMMVAE_ENV_PATH/bin/activate

# Wrap commands passed in from arguments in environment
"$@"

deactivate
