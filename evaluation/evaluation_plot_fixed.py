import pandas as pd
import matplotlib.pyplot as plt

# Define the path for the baseline file (lambda_distil = 0)
data_path = '/home/ymbahram/scratch/baselines/classifier-guidance/results_samesample/data10/FID_KID.csv'


df = pd.read_csv(data_path)

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Get unique g values
unique_g = sorted(df["g"].unique())

# Plot FID over epochs for each g value
for g in unique_g:
    subset = df[df["g"] == g]
    axes[0].plot(subset["epoch"], subset["FID"], label=f"g={g}", marker='o')
axes[0].set_title("FID over Epochs")
axes[0].set_ylabel("FID")
axes[0].legend()
axes[0].grid(True)

# Plot KID over epochs for each g value
for g in unique_g:
    subset = df[df["g"] == g]
    axes[1].plot(subset["epoch"], subset["KID"], label=f"g={g}", marker='o')
axes[1].set_title("KID over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("KID")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.savefig(f'/home/ymbahram/scratch/baselines/classifier-guidance/results_samesample/data10/FID_KID.png')

