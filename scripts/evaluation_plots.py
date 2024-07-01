import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/evaluation.csv'  # Replace with your actual file name
df = pd.read_csv(csv_file)

# Extract column names
columns = df.columns

# Create a figure and subplots
fig, axs = plt.subplots(5, 1, figsize=(10, 15))  # 5 subplots in a single column

# Plot each of the 5 columns against the first column
for i in range(1, 6):
    axs[i-1].plot(df[columns[0]], df[columns[i]], marker='o')
    axs[i-1].set_xlabel(columns[0])
    axs[i-1].set_ylabel(columns[i])
    axs[i-1].set_title(f'{columns[0]} vs {columns[i]}')
    axs[i-1].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/evaluation.png')