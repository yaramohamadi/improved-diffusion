import pandas as pd

# Load the uploaded CSV file to inspect its structure
file_path = '/home/ymbahram/scratch/clf_trg_results/results_samesample/g_p2_a3ft/data10/finetune/FID_KID.csv'
data = pd.read_csv(file_path)

import matplotlib.pyplot as plt

# Define the metric to plot ('FID' or 'KID')
metric = 'FID'  # Change to 'KID' if needed

# Extract unique values for g and gamma
unique_g_values = data['g'].unique()
unique_gamma_values = data['gamma'].unique()

# Set up subplots with a consistent y-axis scale
n_subplots = len(unique_g_values)
fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 6), sharex=True)

if n_subplots == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

# Determine the y-axis limits for consistent scaling
y_min = data[metric].min()
y_max = data[metric].max()

# Create subplots for each g
for ax, g in zip(axes, unique_g_values):
    subset = data[data['g'] == g]
    
    for gamma in unique_gamma_values:
        gamma_subset = subset[subset['gamma'] == gamma]
        ax.plot(gamma_subset['epoch'], gamma_subset[metric], label=f'Gamma: {gamma}')
    
    ax.set_title(f'g = {g}', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.legend(title='Gamma', fontsize=10, loc='best')
    ax.grid(True)
    ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits

# Adjust layout
plt.tight_layout()
plt.savefig('/home/ymbahram/scratch/clf_trg_results/results_samesample/g_p2_a3ft/data10/finetune/FID.png')