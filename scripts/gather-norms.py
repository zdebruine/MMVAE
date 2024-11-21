import tensorflow as tf
import pandas as pd

# Replace with the path to your TensorBoard log file
log_file = "/mnt/projects/debruinz_project/denhofja/cmmvae/lightning_logs/run-experiment/adversarial-conditional.5a9df4a./events.out.tfevents.1730760440.g001.clipper.gvsu.edu.3079995.0"
data = []

for event in tf.compat.v1.train.summary_iterator(log_file):
    for value in event.summary.value:
        # Modify 'grad_norm' to the exact tag name used for gradient norms in your logs
        if "grad_norm" in value.tag:
            data.append(
                {"step": event.step, "grad_norm": value.simple_value, "tag": value.tag}
            )

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("gradient_data.json")
