
# cmmvae.runners CLI and Snakemake Workflow

This README provides instructions on using the command line interface (CLI) included in the `cmmvae` module, along with details on the Snakemake workflow that manages execution depending on necessary resources per rule.

## Command Line Interface (CLI) for cmmvae.runners

### Running the CLI

To invoke the cmmvae CLI, use one the following commands:

```bash
cmmvae --help
Usage: cmmvae [OPTIONS] COMMAND [ARGS]...

  Main entry point for cmmvae CLI

Options:
  --help  Show this message and exit.

Commands:
  autodoc            Build or serve documention automatically using pdoc.
  cli                Run using the LightningCli.
  logger             Logger command group.
  merge-predictions  Merge saved embeddings and metadata into one npz and...
  submit             Submit experiments using configurations from a YAML...
  umap-predictions   Plot UMAP embeddings and optionally log images to...
```

### CLI Commands

## autodoc

```bash
cmmvae autodoc --help
Usage: cmmvae autodoc [OPTIONS] COMMAND [ARGS]...

  Build or serve documention automatically using pdoc.

Options:
  --help  Show this message and exit.

Commands:
  build  Build module documentation to directory with pdoc.
  serve  Serve documention with http server
```
### **:build:**

Builds the module documentation with pdoc placing the output in .cmmvae/docs

After building the documentation all references of '<GIT_REPO_OWNER>' and '<GIT_REPO_NAME>'
are replaced with their respective environemnt variable defenititions.

### **:serve:**

Starts a server http server on port 8000 and starts a command line interface that accepts the following commands:
- `c`|`clean`: Removes all files in .cmmvae/docs
- `b`|`build`: Rebuilds documention

### CLI Features

The CLI offers several options to manage and configure your experiments:

- `--default_root_dir` (str): Specifies the default directory for storing logs and checkpoints.
- `--experiment_name` (str): Defines the name of the experiment directory.
- `--run_name` (str): Specifies the name of the particular experiment run.
- `--predict_dir` (str): Indicates the directory where predictions will be saved after model fitting.

If you do not intend to run a new experiment (e.g., for inference using an existing model checkpoint), you can set the `--run` flag to `False`. In this case, you must also provide:

- `--ckpt_path` (str): Path to the checkpoint file to be loaded by the model.

### Example Command

Here's an example of how you might run the CLI with specific options:

```bash
python -m cmmvae.pipeline.cli \
  --default_root_dir="/path/to/logs" \
  --experiment_name="my_experiment" \
  --run_name="run1" \
  --predict_dir="/path/to/predictions"
```

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
python -m cmmvae.pipeline.cli fit --default_root_dir /path/to/root --experiment_name experiment_name --run_name run_name ...
```

### Model Configuration

By default, Snakemake config expects **`trainer`**, **`model`**, **`data`** as keys and passes those along to the following:

```bash
TRAIN_COMMAND += (
    f" --trainer {config['trainer']} --model {config['model']} "
    f"--data {config['data']} --default_root_dir {ROOT_DIR} "
    f"--experiment_name {EXPERIMENT_NAME} --run_name {RUN_NAME} "
    f"--seed_everything {SEED} "
    f"--predict_dir {PREDICT_SUBDIR} "
)
```

Therefore, to override default config files, you can do something like the following:

```bash
snakemake --profile workflow/profile/slurm --config trainer="config/trainer/config.yaml --trainer.max_epochs 1"
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

By default, Snakemake will create a `.snakemake` directory in the directory where it is run, here you can find snakemake logs. **:When using Slurm profile:**: Inside the main directory, you will find a `.cmmvae` folder with subdirectories for each rule that is executed. These subdirectories contain the `err` and `out` files for each job, named according to the job and the jobId.

### Resource Allocation

- **Training Jobs:** Slurm jobs for training (`train` rule) are configured to run on GPU nodes by default.
- **Merging Predictions and UMAP Generation:** Jobs for merging predictions (`merge_predictions` rule) and generating UMAPs (`umap_predictions` rule) are configured to run on high-memory nodes.

### Running Snakemake

While Snakemake can be executed on the head node, it is recommended to run Snakemake from a submission node for long-running jobs to ensure stability and resource availability.

To test that your setup is working properly you can run the following to perform a quick pass through pipeline:
```bash
sbatch scripts/run-experiment.sh --config trainer=configs/trainer/config.test.yaml experiment_name=testing run_name=quick_test
```
