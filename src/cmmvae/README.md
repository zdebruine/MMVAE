
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
   git clone https://github.com/jdenhof/MMVAE.git
   cd MMVAE
   ```

2. **Set up a virtual environment (optional but recommended):**
*GVSU HPC users can skip

   ```bash
   python -m venv myenv
   source myenv/bin/activate
   pip install .
   ```
   * make sure to set workflow/config.yaml's env_path to the python in virtual env
   * GVSU HPC users can point to /mnt/projects/debruinz_project/pytorch-nightly-env
   and this already taken care of in workflow/config.yaml
   * GVSU HPC venv can be found at /mnt/projects/debruinz_project/pytorch-nightly-env

## Configuration

The workflow requires a configuration file in YAML format, which contains the settings for the experiment. The default location for the config file is `workflow/config.yaml`. You can validate the configuration against a schema defined in `workflow/config.schema.yaml`.

Below is a sample configuration file:

```yaml
root_dir: lightning_logs
experiment_name: default
run_name: default_run
env_path: /mnt/projects/debruinz_project/pytorch-nightly-env/bin/python3
trainer: configs/trainer/config.test.yaml
model: configs/model/config.yaml
data: configs/data/server.yaml
merge_keys:
  - z
categories:
   - 'donor_id'
   - 'assay'
   - 'dataset_id' 
   - 'cell_type'
predict_dir: predictions
umap_dir: umap
seed: 42
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

3. **View Results:**

   Once the workflow completes, results will be stored in the directory specified by `root_dir` within subdirectories for the experiment and run.

## Rules Overview: [Pipeline Documenation](./cmmvae/pipeline.html)

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
