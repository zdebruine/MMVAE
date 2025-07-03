import os
import argparse as ap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

DATA_PATH = "/mnt/projects/debruinz_project/july2024_census_data/subset/"

def main(directory: str):

    output_dir = os.path.join(directory, "distribution")
    df = pd.read_csv(os.path.join(output_dir, "latent_metrics.csv"))

    # Calculate global min and max
    min_val = df.min().min()
    max_val = df.max().max()

    # Round to the nearest 0.5
    ymin = np.floor(min_val * 2) / 2
    ymax = np.ceil(max_val * 2) / 2

    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.ylim([ymin, ymax])
    plt.title("Boxplot of latent-dimension metrics (mean & std) across species")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_metrics.png"))

    df = pd.read_csv(os.path.join(output_dir, "mse.csv"))

    plt.figure(figsize=(12, 6))
    df.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Bar plot of Posterios vs Prior sample")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse.png"))

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    main(args.directory)