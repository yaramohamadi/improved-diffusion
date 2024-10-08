import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Load the CSV file (update the path to your file)
csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/evaluation_KID_intra_LPIPS.csv"
df = pd.read_csv(csv_file)



# Convert the 'intra_LPIPS_dict' column from string to actual dictionary
df['intra_LPIPS_dict'] = df['intra_LPIPS_dict'].apply(ast.literal_eval)

# Extract the unique values of g
unique_g_values = df['g'].unique()

# Get the unique keys across all the rows and create a color map
all_keys = sorted(set().union(*df['intra_LPIPS_dict'].apply(lambda x: x.keys())))
colors = plt.cm.get_cmap('tab10', len(all_keys))  # Get a colormap with enough distinct colors
color_map = {key: colors(i) for i, key in enumerate(all_keys)}

# Number of subplots (one for each unique value of g)
num_subplots = len(unique_g_values)

# Set up the plot layout
fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharex=True)

# If there's only one unique 'g', ensure axes is a list
if num_subplots == 1:
    axes = [axes]

# Iterate through the unique 'g' values and create a subplot for each
for i, g_value in enumerate(unique_g_values):
    # Filter the dataframe for the current g_value
    df_g = df[df['g'] == g_value]

    # Create a DataFrame to store counts for each key at each epoch
    counts_df = pd.DataFrame(0, index=df_g['epoch'], columns=all_keys)

    # Populate the counts DataFrame with values from the dictionaries
    for j, row in df_g.iterrows():
        for key, value in row['intra_LPIPS_dict'].items():
            counts_df.at[row['epoch'], key] = value

    # Apply logarithmic scaling (adding 1 to avoid log(0))
    counts_df = np.log1p(counts_df)

    # Plot the stacked bar chart for the current 'g' value
    bottom = np.zeros(len(counts_df))
    bar_width = 20  # Adjust the width of the bars for thicker bars
    for column in counts_df.columns:
        axes[i].bar(counts_df.index, counts_df[column], bottom=bottom, width=bar_width, color=color_map[column], label=f'Key {column}')
        bottom += counts_df[column]

    # Set titles and labels
    axes[i].set_title(f'Evolution of intra_LPIPS_dict Values Over Epochs (g = {g_value})', fontsize=14)
    axes[i].set_xlabel('Epochs', fontsize=12)
    axes[i].legend(title='Keys', loc='upper right', bbox_to_anchor=(1.15, 1))

# Shared X-axis label
axes[0].set_ylabel('Log-Scaled Count', fontsize=12)

#######################################################################



# # Convert the 'dict' column from string to actual dictionary
# df['intra_LPIPS_dict'] = df['intra_LPIPS_dict'].apply(ast.literal_eval)
# 
# # Extract the unique values of g
# unique_g_values = df['g'].unique()
# 
# # Number of subplots (one for each unique value of g)
# num_subplots = len(unique_g_values)
# 
# # Set up the plot layout
# fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots), sharex=True)
# 
# # If there's only one unique 'g', ensure axes is a list
# if num_subplots == 1:
#     axes = [axes]
# 
# # Iterate through the unique 'g' values and create a subplot for each
# for i, g_value in enumerate(unique_g_values):
#     # Filter the dataframe for the current g_value
#     df_g = df[df['g'] == g_value]
# 
#     # Extract the keys from the dictionary and form the unique set of all keys across the rows
#     unique_keys = sorted(set().union(*df_g['intra_LPIPS_dict'].apply(lambda x: x.keys())))
# 
#     # Create a DataFrame to store counts for each key at each epoch
#     counts_df = pd.DataFrame(0, index=df_g['epoch'], columns=unique_keys)
# 
#     # Populate the counts DataFrame with values from the dictionaries
#     for j, row in df_g.iterrows():
#         for key, value in row['intra_LPIPS_dict'].items():
#             counts_df.at[row['epoch'], key] = value
# 
#     # Apply logarithmic scaling (adding 1 to avoid log(0))
#     counts_df = np.log1p(counts_df)
# 
#     # Plot the stacked area chart for the current 'g' value
#     counts_df.plot(kind='area', stacked=True, cmap='tab10', alpha=0.8, ax=axes[i])
# 
#     # Set titles and labels
#     axes[i].set_title(f'Evolution of intra_LPIPS_dict Values Over Epochs (g = {g_value})', fontsize=14)
#     axes[i].set_ylabel('Log-Scaled Count', fontsize=12)
#     axes[i].legend(title='Keys', loc='upper right', bbox_to_anchor=(1.15, 1))
# 
# # Shared X-axis label
# axes[-1].set_xlabel('Epochs', fontsize=12)

# Show the plot
plt.tight_layout()
plt.savefig('/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/intra_LPIPS_dict.png')