
# Snakemake Workflow for MMVAE Pipeline

This repository provides a Snakemake workflow for running a pipeline to train and evaluate a conditional multi-modal variational autoencoder (cmmVAE). This pipeline performs training, merges predictions, and generates UMAP visualizations.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Rules Overview](#rules-overview)
- [Further Information](#further-information)

## Installation

To use this workflow, you must have Snakemake and the required dependencies installed on your system. Follow these steps to set up the environment:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/<GIT_REPO_OWNER>/<GIT_REPO_NAME>.git
    cd <GIT_REPO_NAME>
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -e .[test,doc]
    ```

## Configuration

The workflow requires a configuration file in YAML format, which contains the settings for the experiment. The default location for the config file is `workflow/config.yaml`. You can validate the configuration against a schema defined in `workflow/config.schema.yaml`.

Below is a sample configuration file:

```yaml
root_dir: lightning_logs

config_name: config.yaml
experiment_name: experiment
run_name: version000

predict_dir: samples

train_command: fit --trainer configs/trainer/config.test.yaml --model configs/model/config.yaml --data configs/data/local.yaml

categories:
- 'donor_id'
- 'assay'
- 'dataset_id'
- 'cell_type'

merge_keys:
- z
```

## Usage

To run the Snakemake workflow, follow these steps:

1. **Edit the Configuration File:**

   Modify `workflow/config.yaml` to match your experiment's needs. Ensure that all paths and parameters are set correctly.<br>
   Default model, data, and trainer configurations are stored in `configs` and can be modified or overriden.<br>

   Look at [LightingCli Documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) for more information on how to configure the yaml files.<br>
   The cmmvae cli automatically runs the models predict step after fit subcommand.

2. **Run Snakemake:**

   Execute the following command to run the workflow on the head node:

   ```bash
      snakemake --profile workflow/profile/slurm
   ```

   Optionally, add `--config option=value` to override any config file parameter.

   **SLURM USERS**
   ```bash
   export CMMVAE_ENV_PATH="/path/to/env"
   sbatch scripts/run-snakemake.sh
   ```

3. **View Results:**

   Once the workflow completes, results will be stored in the directory specified by `root_dir` within subdirectories for the experiment and run.

4. **Batch Experimentation**

   The `cmmvae submit` command allows for batch experimentation based on a experiments.yaml file. The schema of this yaml is identical to that of the workflow/config.schema.yaml with the exception that the train_command key is configurable. Keys found in experiments.yaml with override the defaults in workflow/config.yaml. The train_command is a dictionary with a single key value pair representing the subcommand to pass to the LightningCli and the value is a dictionary of options.

   The options are parsed as either their key value appending -- to each key for the cli command or if the value is a dictionary where the only is key is "track" then the value is a dictionary of name values to track. The name is appened to the run_name by dot notation.

   Below is an example experiments.yaml file that will kick off two experiments.

```yaml

experiment_name: experiment
run_name: version000

train_command:
  fit:
    model: configs/model/config.yaml
    trainer:
      track:
        test: configs/trainer/config.test.yaml
        full: configs/trainer/config.yaml
    data: configs/data/local.yaml

categories:
- 'donor_id'
- 'assay'
- 'dataset_id'
- 'cell_type'

merge_keys:
- z
```

## Tracking Experiments

* **:Preview last experiment:** cmmvae logger last
* **:Preview experiment history:** cmmvae logger history
* **:Help:** cmmvae --help

## Rules Overview: [Pipeline Documenation](./cmmvae/runners.html)

This Snakemake workflow consists of the following rules:

1. **`all`:** The default rule that specifies the final target files (evaluation files) that should be produced by the workflow.

2. **`train`:** Trains the cmmVAE model using the specified configuration and saves the model checkpoint and predictions.

3. **`merge_predictions`:** Merges the prediction files generated during training into a unified format for further analysis.

4. **`umap_predictions`:** Generates UMAP visualizations based on the merged predictions and saves the resulting images.

## Further Information

Snakemake requires that rule's complete gracefully and at the end of the rule the file's specified by it's output exist. If they do not
they will be deleted unless --keep-incomplete specified.

For additional information on Snakemake, consult the [Snakemake documentation](https://snakemake.readthedocs.io/).

If you encounter any issues or have questions, feel free to reach out via the repository's issue tracker or contact the maintainers directly.
