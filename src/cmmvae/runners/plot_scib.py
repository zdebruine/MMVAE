import pandas as pd
import matplotlib.pyplot as plt

PATH = "/mnt/projects/debruinz_project/july2024_census_data/subset/"

results1 = pd.read_csv("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/cycle_consistency/batch_norm_cycle/scib_metrics.csv")
results2 = pd.read_csv("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/cycle_consistency/batch_norm_no_cycle/scib_metrics.csv")

df = pd.merge(
    results1.reset_index(),
    results2.reset_index(),
    on='metric',
    suffixes=('_run1', '_run2')
)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['score_run1'], df['score_run2'], s=50, edgecolor='k', alpha=0.7)

# Annotate each point with the metric name
for i, m in enumerate(df['metric']):
    plt.annotate(m,
                 (df['score_run1'].iloc[i], df['score_run2'].iloc[i]),
                 textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

plt.xlabel('Score (batch_norm_cycle)')
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
plt.ylabel('Score (batch_norm_no_cycle)')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
plt.title('SCIB Metrics: No Cycle vs. Cycle')
plt.grid(True)
plt.axline((0, 0), slope=1, color='r', linestyle='--', label='x=y')
plt.tight_layout()
plt.savefig("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/cycle_consistency/batch_norm_cycle/scib_metrics.png")
plt.savefig("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/cycle_consistency/batch_norm_no_cycle/scib_metrics.png")