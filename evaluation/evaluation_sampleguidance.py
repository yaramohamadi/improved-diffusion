import pandas as pd
import matplotlib.pyplot as plt

# File paths for the CSVs
file_paths = ['/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_and_sampling/0_1/evaluation_0_1.csv', 
              '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_and_sampling/0_1/evaluation_0_05.csv', 
              '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_and_sampling/0_1/evaluation_0.csv']

# Read the CSVs into pandas DataFrames
df_0_1 = pd.read_csv(file_paths[0])
df_0_05 = pd.read_csv(file_paths[1])
df_0 = pd.read_csv(file_paths[2])

# Extract the FID column
fid_0_1 = df_0_1['FID']
fid_0_05 = df_0_05['FID']
fid_0 = df_0['FID']

# Define the x values as 0, 25, 50, 75, ..., based on the number of rows
x_values = [i * 25 for i in range(len(fid_0))]

# Plot the FID values from the three CSVs
plt.plot(x_values, fid_0_1, label='FID - 0.1')
plt.plot(x_values, fid_0_05, label='FID - 0.05')
plt.plot(x_values, fid_0, label='FID - 0')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('FID')
plt.title('FID Comparison Across Different Sampling Guidances (Trained on 0.1 fixed guidance)')

# Show legend
plt.legend()

# Show the plot
plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_and_sampling/0_1/FID.png')