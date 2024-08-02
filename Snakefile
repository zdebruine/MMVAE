# Snakefile
from snakemake.utils import validate
import os

# Load the config file
configfile: "workflow/config.yaml"

# Validate the config file
validate(config, "workflow/config.schema.yaml")

# Extract configuration details with defaults and checks
ROOT_DIR = config["root_dir"]
EXPERIMENT_NAME = config["experiment_name"]
RUN_NAME = config.get("run_name", "default_run")
RUN_DIR = os.path.join(ROOT_DIR, EXPERIMENT_NAME, RUN_NAME)

ENV_PATH = config["env_path"]
CONFIG_NAME = config.get("config_name", "config.yaml")
CONFIG_DIR = config.get("config_dir", None)

MERGE_KEYS = config["merge_keys"]
CATEGORIES = config.get("categories", [])
PREDICT_SUBDIR = config.get("predict_dir", "predictions")
PREDICT_DIR = os.path.join(RUN_DIR, PREDICT_SUBDIR)

UMAP_PATH = config.get("umap_dir", "umap")
UMAP_DIR = os.path.join(RUN_DIR, UMAP_PATH)

# Separate directory for merged outputs to avoid conflicts
MERGED_DIR = os.path.join(RUN_DIR, "merged")

EMBEDDINGS_PATHS = [os.path.join(MERGED_DIR, f"{key}_embeddings.npz") for key in MERGE_KEYS]
METADATA_PATHS = [os.path.join(MERGED_DIR, f"{key}_metadata.pkl") for key in MERGE_KEYS]

TRAIN_CONFIG_FILE = os.path.join(RUN_DIR, CONFIG_NAME)
CKPT_PATH = os.path.join(RUN_DIR, "checkpoints", "best_model.ckpt")

SEED = config.get('seed', False)

EVALUATION_FILES = expand(
    "{root_dir}/{experiment}/{run}/{results}/integrated.{category}.umap.{key}.png",
    root_dir=ROOT_DIR,
    experiment=EXPERIMENT_NAME,
    run=RUN_NAME,
    category=CATEGORIES,
    results=UMAP_PATH,
    key=MERGE_KEYS,
)

TRAIN_COMMAND = f"-m sciml fit"
if CONFIG_DIR:
    TRAIN_COMMAND += f" -c {CONFIG_DIR}"
else:
    TRAIN_COMMAND += (
        f" --trainer {config['trainer']} --model {config['model']} "
        f"--data {config['data']} --default_root_dir {ROOT_DIR} "
        f"--experiment_name {EXPERIMENT_NAME} --run_name {RUN_NAME} "
        f"--seed_everything {SEED} "
        f"--predict_dir {PREDICT_SUBDIR} "
    )

rule all:
    input:
        EVALUATION_FILES

rule train:
    output:
        config_file=TRAIN_CONFIG_FILE,
        ckpt_path=CKPT_PATH,
        predict_dir=directory(PREDICT_DIR)
    params:
        env_path=ENV_PATH,
        command=TRAIN_COMMAND,
    shell:
        """
        {params.env_path} {params.command}
        """

rule merge_predictions:
    input: 
        predict_dir=PREDICT_DIR,
    output:
        embeddings_path=EMBEDDINGS_PATHS,
        metadata_path=METADATA_PATHS,
    params:
        env_path=ENV_PATH,
        merge_keys=" ".join(MERGE_KEYS),
    shell:
        """
        mkdir -p {MERGED_DIR}
        {params.env_path} -m sciml.pipeline.merge_predictions --directory {input.predict_dir} --keys {params.merge_keys} --save_dir {MERGED_DIR}
        """

rule umap_predictions:
    input:
        embeddings_path=EMBEDDINGS_PATHS,
        metadata_path=METADATA_PATHS,
    output:
        EVALUATION_FILES,
    params:
        env_path=ENV_PATH,
        predict_dir=MERGED_DIR,
        save_dir=UMAP_DIR,
        categories=" ".join(CATEGORIES),
        merge_keys=" ".join(MERGE_KEYS),
    shell:
        """
        {params.env_path} -m sciml.pipeline.generate_umap --directory {params.predict_dir} --save_dir {params.save_dir} --categories {params.categories} --keys {params.merge_keys}
        """