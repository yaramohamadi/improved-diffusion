import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/ymbahram/scratch/baselines_avg/classifier-guidance/data10/FID_KID.csv'
data = pd.read_csv(file_path)

# Group by gamma and epoch, then calculate mean and std for FID
grouped = data.groupby(['gamma', 'epoch']).agg(
    mean_FID=('FID', 'mean'),
    std_FID=('FID', 'std')
).reset_index()

# Convert columns to numeric to avoid errors
grouped['epoch'] = pd.to_numeric(grouped['epoch'], errors='coerce')
grouped['mean_FID'] = pd.to_numeric(grouped['mean_FID'], errors='coerce')
grouped['std_FID'] = pd.to_numeric(grouped['std_FID'], errors='coerce')

# Drop any rows with NaN values
grouped = grouped.dropna(subset=['epoch', 'mean_FID', 'std_FID'])

# Plot the data
plt.figure(figsize=(10, 6))

for gamma, group in grouped.groupby('gamma'):
    plt.plot(group['epoch'], group['mean_FID'], label=f"Gamma: {gamma}")
    plt.fill_between(
        group['epoch'], 
        group['mean_FID'] - group['std_FID'], 
        group['mean_FID'] + group['std_FID'], 
        alpha=0.2
    )

# Customize the plot
plt.title("Average FID vs Epochs with Standard Deviation")
plt.xlabel("Epoch")
plt.ylabel("Average FID")
plt.legend(title="Gamma")
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig("/home/ymbahram/scratch/baselines_avg/classifier-guidance/data10/FID_KID.png")