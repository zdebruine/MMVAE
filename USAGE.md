
# CLI and Snakemake Workflow

The CMMVAE comes with a cli that contains commands that are useful for submitting and monitoring
experiments. Simply run `cmmvae --help` for more information.

## Table of Contents

1. [Running the CLI](#running-the-cli)
2. [Submiting Experiments](#submiting-experiments)
3. [Logging and Monitoring](#logging-and-monitoring)
4. [Snakemake Workflow](#snakemake-workflow)

## Running the CLI

To invoke the cmmvae CLI, use one the following commands:

```bash
cmmvae --help
Usage: cmmvae [OPTIONS] COMMAND [ARGS]...

  Main entry point for cmmvae CLI

Options:
  --help  Show this message and exit.

Commands:
  logger    Logger command group.
  submit    Submit experiments using configurations from a YAML file.
  workflow  Workflow commands for experiments.
```

## Submiting Experiments

Generates combinations of experiments to run. By default, configuration
for experiments is at experiments.yaml and can be overriden with --config_file.
All options specified with --config are parsed as keyword arguments as `key=value` (ie. `--config key=value --config key2=value2`).
If no key values are present in the config_file then default snakemake config applys.

To configure multiple experiments at a time replace the value that changes with a dictionary of a single key `track` which
has a dictionary of either a single key and value of list or dictionary name value pairs. The names are appened to the run_name
with dot notation.

```yaml
run_name: default
train_command:
  fit:
    model: configs/model/config.yaml
    trainer:
      track:
        test: configs/trainer/config.test.yaml
        # full: configs/trainer/config.yaml
    data:
      track:
        local: configs/data/local.yaml
        server: configs/data/server.yaml
```

The experiments.yaml configuration provided will create two experiments. One with the train_command:
```
fit --model configs/model/config.yaml --trainer configs/trainer/config.test.yaml --data configs/data/local.yaml --run_name=default.test.local
```
and the other:
```
fit --model configs/model/config.yaml --trainer configs/trainer/config.test.yaml --data configs/data/server.yaml --run_name=default.test.server
```

```bash
cmmvae submit --help
Usage: cmmvae submit [OPTIONS]

  Submit experiments using configurations from a YAML file.

  Args:     config_file (str): Path to the YAML configuration file.
  max_job_limit (int): Maximum number of jobs that can be run.     preview
  (bool): Preview job configurations without running them.

Options:
  --config_file TEXT           Path to configuration file.  [default:
                               experiments.yaml]
  --config TEXT                Configuration options as key=value pairs
  -m, --max_job_limit INTEGER  Max number of jobs capable of outputting
                               without failure.  [default: 10]
  -t, --timestamp              Added timestamp to end of run name.
  -p, --preview                Do not run subprocess, only preview job
                               configurations.
  --help                       Show this message and exit.
```

## Logging and Monitoring

Review stdout and stderr files from previous snakemake submissions.

```bash
cmmvae logger --help
Usage: cmmvae logger [OPTIONS] COMMAND [ARGS]...

  Logger command group.

Options:
  --help  Show this message and exit.

Commands:
  history  Display the last n jobs in history.
  job      Display the job by submission jobid
  last     View the last job or a specified job.
```

# Snakemake Workflow

Workflow commands responsible for training, running inference, and visualizing predictions.

```bash
cmmvae workflow --help
Usage: cmmvae workflow [OPTIONS] COMMAND [ARGS]...

  Workflow commands for experiments.

Options:
  --help  Show this message and exit.

Commands:
  cli                Run using the LightningCli.
  merge-predictions  Merge saved embeddings and metadata into one npz and...
  umap-predictions   Plot UMAP embeddings and optionally log images to...
```


### cli: Features

The CLI offers several options to manage and configure your experiments:

- `--default_root_dir` (str): Specifies the default directory for storing logs and checkpoints.
- `--experiment_name` (str): Defines the name of the experiment directory.
- `--run_name` (str): Specifies the name of the particular experiment run.
- `--predict_dir` (str): Indicates the directory where predictions will be saved after model fitting.

If you do not intend to run a new experiment (e.g., for inference using an existing model checkpoint), you can set the `--run` flag to `False`. In this case, you must also provide:

- `--ckpt_path` (str): Path to the checkpoint file to be loaded by the model.

### Additional Functionality

- **Before Fit**: The CLI prints the model configuration and saves its hyperparameters before beginning the fitting process.
- **After Fit**: Upon completion of model fitting, the CLI automatically executes the predict subcommand using the best checkpoint found during training.

This CLI is designed to provide a seamless and user-friendly interface for conducting experiments, managing configurations, and logging results efficiently.

## Snakemake Workflow

Snakemake is responsible for managing execution depending on necessary resources per rule. The Snakemake pipeline for CMMVAE follows the following rules:

### Rules Overview

This Snakemake workflow consists of the following rules:

1. **`all`:** The default rule that specifies the final target files (evaluation files) that should be produced by the workflow.

2. **`train`:** Trains the cmmVAE model using the specified configuration and saves the model checkpoint and predictions.

3. **`merge_predictions`:** Merges the prediction files generated during training into a unified format for further analysis.

4. **`umap_predictions`:** Generates UMAP visualizations based on the merged predictions and saves the resulting images.

### Rules

#### all

The last rule that specifies completion of all jobs.

#### train

Train the model and output a config.yaml to reinitialize state and the best model checkpoint.

In order to train models separately from Snakemake, run the following:

```bash
cmmvae workflow cli fit -c config/config.yaml --default_root_dir /path/to/root --experiment_name experiment_name --run_name run_name ...
```

### Model Configuration

By default, Snakemake config passes those along by the following:

```bash
TRAIN_COMMAND += (
    f"--default_root_dir {ROOT_DIR} "
    f"--experiment_name {EXPERIMENT_NAME} "
    f"--run_name {RUN_NAME} "
    f"--seed_everything {SEED} "
    f"--predict_dir {PREDICT_SUBDIR} "
)
```

Therefore, to override default config files, you can do something like the following:

```bash
snakemake --profile workflow/profile/slurm --config train_command="fit -c ..." --run_name="version000"
```

In the config.yaml files, you can specify class configurations by using the following structure:

```yaml
class_path: path.to.Class
init_args:
    arg1: argument1
    arg2: 2
```

Look to [LightningCLI Documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html) for more resources on configuring YAML files.

## Configuring Workflow Execution

To configure the workflow execution, you can modify the configuration file located at `workflow/profile/slurm`. This file contains the default settings for cluster execution, including where logs are stored.

### Log Management

By default, Snakemake will create a `.snakemake` directory in the directory where it is run, here you can find snakemake logs. **When using Slurm profile**: Inside the main directory, you will find a `.cmmvae` folder with subdirectories for each rule that is executed. These subdirectories contain the `err` and `out` files for each job, named according to the job and the jobId.

### Resource Allocation

Workflow resource allocation is determined by the rule executed and the configuration provided in workflow/profile/slurm/config.yaml.

- **Training Jobs:** Slurm jobs for training (`train` rule) are configured to run on GPU nodes by default.
- **Merging Predictions and UMAP Generation:** Jobs for merging predictions (`merge_predictions` rule) and generating UMAPs (`umap_predictions` rule) are configured to run on high-memory nodes.

### Running Snakemake

While Snakemake can be executed on the head node, it is recommended to run Snakemake from a submission node for long-running jobs to ensure stability and resource availability.

To test that your setup is working properly you can run the following to perform a quick pass through pipeline:

Run a single experiment based off of workflow/config.yaml:
```bash
sbatch scripts/run-experiment.sh --config trainer=configs/trainer/config.test.yaml --config experiment_name=testing --config run_name=quick_test
```
**or**
Run single or multiple experiments based off experiments.yaml:
```bash
cmmvae submit -t
```
