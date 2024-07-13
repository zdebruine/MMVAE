from snakemake.utils import validate
import os
import yaml
import warnings
import glob

# Load the config file
configfile: "workflow/config.yaml"

# Validate the config file (uncomment if validation schema is available)
validate(config, "workflow/config.schema.yaml")

ROOT_DIR = config["root_dir"]
EXPERIMENT_NAME = config["experiment_name"]
RUN_NAME = config["run_name"]

RUN_DIR = os.path.join(ROOT_DIR, EXPERIMENT_NAME, RUN_NAME)

if config.get('retrying', False):
    if not os.path.exists(RUN_DIR):
        raise RuntimeError(f"{RUN_DIR} is not a directory")

SM_CONFIG_FILE = os.path.join(RUN_DIR, "sm_config.yaml")

if os.path.exists(SM_CONFIG_FILE) and not config.get("override", False):
    warnings.warn(f"Overriding supplied configfile with {SM_CONFIG_FILE}")
    with open(SM_CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
else:
    warnings.warn(f"Setting new config to {SM_CONFIG_FILE}")
    os.makedirs(os.path.dirname(SM_CONFIG_FILE), exist_ok=True)
    with open(SM_CONFIG_FILE, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

LIGHTNING_CONFIG_PATH = os.path.join(RUN_DIR, config["config_name"])  # config.yaml
CKPT_PATH = os.path.join(RUN_DIR, config["ckpt_path"])  # relative to run_dir

EMBEDDINGS_PATTERN = [os.path.join(RUN_DIR, path) for path in config['embeddings']]
METADATA_PATTERN = [os.path.join(RUN_DIR, path) for path in config['metadata']]

EVALUATION_FILES = [f"{RUN_DIR}/results/integrated.{category}.umap.{os.path.basename(key).removesuffix('_embeddings.npz')}.png" for category in ['assay', 'cell_type', 'dataset_id'] for key in config['embeddings']]

ENV_PATH = config["env_path"]

rule all:
    input:
        EVALUATION_FILES

rule train:
    output:
        config_file = LIGHTNING_CONFIG_PATH,
        ckpt_path = CKPT_PATH
    params:
        lighting_fit_args = config.get('lightning_fit_args', "")
    shell:
        """
        {ENV_PATH} -m sciml fit {params.lighting_fit_args} --default_root_dir {ROOT_DIR} --experiment_name {EXPERIMENT_NAME} --run_name {RUN_NAME}
        """

rule integrate:
    input:
        config_file = LIGHTNING_CONFIG_PATH,
        ckpt_path = CKPT_PATH
    output:
        embeddings = EMBEDDINGS_PATTERN,
        metadata = METADATA_PATTERN,
        
    shell:
        """
        {ENV_PATH} -m sciml.pipeline.integrate -c {input.config_file} --ckpt_path {input.ckpt_path}
        """

rule umap:
    input:
        embeddings = EMBEDDINGS_PATTERN,
        metadata = METADATA_PATTERN,
    output:
        EVALUATION_FILES
    shell:
        """
        {ENV_PATH} -m sciml.pipeline.generate_umap --directory {RUN_DIR} --embedding_paths {input.embeddings} --metadata_paths {input.metadata}
        """