## Snakemake workflow for the CMMVAE (Conditional Multimodal Variational Autoencoder) module.
## This workflow automates the training, merging, and evaluation processes for CMMVAE models.

## Import necessary libraries and modules.
from snakemake.utils import validate
import os

## Load the configuration file specified for this workflow.
configfile: "workflow/config.yaml"

## Validate the configuration file against a predefined schema to ensure correctness and completeness.
validate(config, "workflow/config.schema.yaml")

## Define the root directory for the experiment, extracted from the configuration file.
## This directory serves as the main folder where all experiment-related outputs will be stored.
ROOT_DIR = config["root_dir"]

## Define the name of the experiment, extracted from the configuration file.
## This name is used to distinguish between different experiments within the root directory.
EXPERIMENT_NAME = config["experiment_name"]

## Define the name of the specific run within the experiment. If not provided, defaults to "default_run".
RUN_NAME = config.get("run_name", "default_run")

## Define the directory path where this specific run's results will be stored.
RUN_DIR = os.path.join(ROOT_DIR, EXPERIMENT_NAME, RUN_NAME)

## Define the name of the configuration file specific to this run. Defaults to "config.yaml" if not specified.
CONFIG_NAME = config.get("config_name", "config.yaml")

## Define the directory containing the configuration file. If not provided, assumes the default structure.
CONFIG_DIR = config.get("config_dir", None)

## Define the keys that are used for merging results from different experiments.
MERGE_KEYS = config["merge_keys"]

## Define the categories for which UMAP visualizations will be generated.
## This is optional, and if not provided, defaults to an empty list.
CATEGORIES = config.get("categories", [])

## Define the subdirectory where prediction results will be stored. Defaults to "predictions".
PREDICT_SUBDIR = config.get("predict_dir", "predictions")

## Define the full directory path for storing predictions within the run directory.
PREDICT_DIR = os.path.join(RUN_DIR, PREDICT_SUBDIR)

## Define the directory where UMAP visualizations will be saved.
## Defaults to "umap" within the run directory.
UMAP_PATH = config.get("umap_dir", "umap")
UMAP_DIR = os.path.join(RUN_DIR, UMAP_PATH)

## Define the directory where meta discriminator tensorboard files will be saved.
## Defaults to "meta_disc" within the run directory.
META_DISC_PATH = config.get("meta_disc_dir", "meta_disc")
META_DISC_DIR = os.path.join(RUN_DIR, META_DISC_PATH)

## Define a separate directory for merged outputs to avoid conflicts between different merge operations.
MERGED_DIR = os.path.join(RUN_DIR, "merged")

## Define the directory to store correlation outputs
CORRELATION_PATH = config.get("correlation_dir", "correlations")
CORRELATION_DIR = os.path.join(RUN_DIR, CORRELATION_PATH)

CORRELATION_DATA = config["correlation_data"]

## Generate the paths for the embeddings and metadata files based on the merge keys.
EMBEDDINGS_PATHS = [os.path.join(MERGED_DIR, f"{key}_embeddings.npz") for key in MERGE_KEYS]
METADATA_PATHS = [os.path.join(MERGED_DIR, f"{key}_metadata.pkl") for key in MERGE_KEYS]

## Define the path to the training configuration file within the run directory.
TRAIN_CONFIG_FILE = os.path.join(RUN_DIR, CONFIG_NAME)

## Define the path to the checkpoint file for saving the best model.
CKPT_PATH = os.path.join(RUN_DIR, "checkpoints", "best_model.ckpt")

## Optional: Set a seed value for reproducibility. If not specified, the seed is set to False.
## This ensures that the results can be replicated exactly in subsequent runs.
SEED = config.get('seed', False)

## Generate file paths for UMAP evaluation images using the configured directory structure.
EVALUATION_FILES = expand(
    "{root_dir}/{experiment}/{run}/{results}/integrated.{category}.umap.{key}.png",
    root_dir=ROOT_DIR,
    experiment=EXPERIMENT_NAME,
    run=RUN_NAME,
    category=CATEGORIES,
    results=UMAP_PATH,
    key=MERGE_KEYS,
)

MD_FILES = expand(
    "{md_logs}/events.out.tfevents.*",
    md_logs=META_DISC_DIR

)

## Construct the command to run the CMMVAE training pipeline.
## If a configuration directory is provided, it is included in the command; otherwise,
## individual parameters such as trainer, model, and data are passed explicitly.
TRAIN_COMMAND = config["train_command"]
CORRELATION_COMMAND = config["correlation_command"]

TRAIN_COMMAND += str(
    f" --default_root_dir {ROOT_DIR} "
    f"--experiment_name {EXPERIMENT_NAME} --run_name {RUN_NAME} "
    f"--seed_everything {SEED} "
    f"--predict_dir {PREDICT_SUBDIR} "
)

CORRELATION_COMMAND += str(
    f" --default_root_dir {ROOT_DIR} "
    f"--experiment_name {EXPERIMENT_NAME} --run_name {RUN_NAME} "
    f"--seed_everything {SEED} "
    f"--predict_dir {PREDICT_SUBDIR} "
    f"--ckpt_path {CKPT_PATH} "
)

CORRELATION_FILES = expand(
    "{correlation_dir}/correlations.csv",
    correlation_dir=CORRELATION_DIR,
)

CORRELATION_FILES += expand(
    "{correlation_dir}/correlations.pkl",
    correlation_dir=CORRELATION_DIR,
)

## Define the final output rule for Snakemake, specifying the target files that should be generated
## by the end of the workflow.
rule all:
    input:
        EVALUATION_FILES,
        CORRELATION_FILES
        # MD_FILES

## Define the rule for training the CMMVAE model.
## The output includes the configuration file, the checkpoint path, and the directory for predictions.
rule train:
    output:
        config_file=TRAIN_CONFIG_FILE,
        ckpt_path=CKPT_PATH,
        predict_dir=directory(PREDICT_DIR)
    params:
        command=TRAIN_COMMAND
    shell:
        """
        cmmvae workflow cli {params.command}
        """

## Define the rule for merging predictions.
## This rule takes the prediction directory as input and outputs the embeddings and metadata files.
rule merge_predictions:
    input:
        predict_dir=PREDICT_DIR,
    output:
        embeddings_path=EMBEDDINGS_PATHS,
        metadata_path=METADATA_PATHS,
    params:
        merge_keys=" ".join(MERGE_KEYS),
    shell:
        """
        mkdir -p {MERGED_DIR}
        cmmvae workflow merge-predictions --directory {input.predict_dir} --keys {params.merge_keys} --save_dir {MERGED_DIR}
        """

## Define the rule for getting R^2 correlations on the filtered data
## This rule outputs correlation scores per filtered data group
rule correlations:
    input:
        embeddings_path=EMBEDDINGS_PATHS,
    output:
        CORRELATION_FILES,
    params:
        command=CORRELATION_COMMAND,
        data=CORRELATION_DATA,
        save_dir=CORRELATION_DIR,
    shell:
        """
        mkdir -p {CORRELATION_DIR}
        cmmvae workflow correlations {params.command} --correlation_data {params.data} --save_dir {params.save_dir}
        """

## Define the rule for generating UMAP visualizations from the merged predictions.
## This rule produces UMAP images for each combination of category and merge key.
rule umap_predictions:
    input:
        embeddings_path=EMBEDDINGS_PATHS,
        metadata_path=METADATA_PATHS,
    output:
        EVALUATION_FILES,
    params:
        predict_dir=MERGED_DIR,
        save_dir=UMAP_DIR,
        categories=" ".join(f"--categories {category}" for category in CATEGORIES),
        merge_keys=" ".join(f"--keys {merge_key}" for merge_key in MERGE_KEYS),
    shell:
        """
        cmmvae workflow umap-predictions --directory {params.predict_dir} {params.categories} {params.merge_keys} --save_dir {params.save_dir}
        """

# rule meta_discriminators:
#     input:
#         CKPT_PATH
#     output:
#         MD_FILES,
#     params:
#         log_dir=META_DISC_DIR,
#         ckpt=CKPT_PATH,
#         config=TRAIN_CONFIG_FILE
#     shell:
#         """
#         cmmvae workflow meta-discriminator --log_dir {params.log_dir} --ckpt {params.ckpt} --config {params.config}
#         """