import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file  
file_path = '/home/ymbahram/scratch/baselines/classifier-guidance/data10/FID_KID.csv'
df = pd.read_csv(file_path)

# Get the unique values of 'g' to create subplots for each
unique_g_values = df['g'].unique()

# Create subplots
num_subplots = len(unique_g_values)
fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4 * num_subplots), sharex=True)

# Plot FID over epochs for each unique g value
for i, g_value in enumerate(unique_g_values):
    subset = df[df['g'] == g_value]
    for gamma in subset['gamma'].unique():
        gamma_subset = subset[subset['gamma'] == gamma]
        axes[i].plot(gamma_subset['epoch'], gamma_subset['FID'], label=f'Gamma={gamma}')
    
    axes[i].set_title(f'FID over Epochs for g={g_value}')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('FID')
    axes[i].legend()
    axes[i].grid(True)
    axes[i].set_ylim([50,180])

plt.tight_layout()

# Show the plot
plt.savefig("/home/ymbahram/scratch/baselines/classifier-guidance/data10/FID_KID.png")