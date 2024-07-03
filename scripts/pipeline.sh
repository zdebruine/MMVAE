#!/bin/bash

default_root_dir=$2
experiment_name=$3
run_name=$4

RUN_DIRECTORY=$default_root_dir/$experiment_name/$run_name

MODULE_PATH=/mnt/projects/debruinz_project/denhofja/sciml

# Shift the positional parameters to the left by 4
shift 4

# Capture the remaining command-line arguments for the training job
training_args="$@"

# Job 1: Train Model
train_script=$(mktemp)
cat << EOF > $train_script
#!/bin/bash
source /mnt/projects/debruinz_project/pytorch-nightly-env/bin/activate
python -m sciml $training_args --default_root_dir $default_root_dir --experiment_name $experiment_name --run_name $run_name
EOF
chmod +x $train_script

jid1=$(sbatch --job-name=fitting \
     --output=job.%j.out \
     --error=job.%j.err \
     --time=1-00:00:00 \
     --partition=gpu \
     --gpus-per-node=tesla_v100s:1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --mem=179G \
     --chdir=$chdir \
     --export=ALL,PYTHONPATH=$MODULE_PYTHON_PATH:$PYTHONPATH \
     $train_script \
     --parsable)

if [ $? -ne 0 ]; then
    echo "Job 1 submission failed"
    exit 1
else
    echo "Job 1 submitted successfully with Job ID: $jid1"
fi

# Ensure jid1 is a valid job ID by extracting the numeric part
jid1=$(echo $jid1 | awk '{print $NF}')

# Job 2: Get Predictions from Model
predict_script=$(mktemp)
cat << EOF > $predict_script
#!/bin/bash
source /mnt/projects/debruinz_project/pytorch-nightly-env/bin/activate
python -m sciml.pipeline.generate_predictions --directory $RUN_DIRECTORY
EOF
chmod +x $predict_script

jid2=$(sbatch --job-name=predicting \
     --output=job.%j.out \
     --error=job.%j.err \
     --time=1-00:00:00 \
     --partition=gpu \
     --ntasks=1 \
     --gpus-per-node=tesla_v100s:1 \
     --cpus-per-task=6 \
     --mem=179G \
     --dependency=afterok:$jid1 \
     --chdir=$chdir \
     --export=ALL,PYTHONPATH=$MODULE_PYTHON_PATH:$PYTHONPATH \
     $predict_script \
     --parsable)

if [ $? -ne 0 ]; then
    echo "Job 2 submission failed"
    exit 1
else
    echo "Job 2 submitted successfully with Job ID: $jid2"
fi

# Ensure jid2 is a valid job ID by extracting the numeric part
jid2=$(echo $jid2 | awk '{print $NF}')

# Job 3: Processing results from Model
umap_script=$(mktemp)
cat << EOF > $umap_script
#!/bin/bash
source /mnt/projects/debruinz_project/pytorch-nightly-env/bin/activate
python -m sciml.pipeline.generate_umap --directory $RUN_DIRECTORY
EOF
chmod +x $umap_script

sbatch --job-name=umap \
     --output=job.%j.out \
     --error=job.%j.err \
     --time=2:00:00 \
     --partition=all \
     --ntasks=1 \
     --cpus-per-task=40 \
     --mem=179G \
     --dependency=afterok:$jid2 \
     --chdir=$chdir \
     --export=ALL,PYTHONPATH=$MODULE_PYTHON_PATH:$PYTHONPATH \
     $umap_script

if [ $? -ne 0 ]; then
    echo "Job 3 submission failed"
    exit 1
else
    echo "Job 3 submitted successfully"
fi
