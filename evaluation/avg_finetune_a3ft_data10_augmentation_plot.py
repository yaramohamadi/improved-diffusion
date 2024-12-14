import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv(f"/home/ymbahram/scratch/baselines_avg/a3ft/data10/augmentation_FID_KID.csv")


df['augmentation'] = df['augmentation'].fillna('None')

# Group data by 'augmentation' and 'epoch', calculate mean and std for FID
grouped = df.groupby(['augmentation', 'epoch'])['FID'].agg(['mean', 'std']).reset_index()

# Create the plot
plt.figure(figsize=(10, 6))

# Loop through each unique augmentation
for augmentation in grouped['augmentation'].unique():

    print(augmentation)
    data = grouped[grouped['augmentation'] == augmentation]
    epochs = data['epoch']
    mean_fid = data['mean']
    std_fid = data['std']
    
    # Plot the mean FID
    plt.plot(epochs, mean_fid, label=f'Augmentation: {augmentation}')
    
    # Fill the area between mean ± std
    plt.fill_between(epochs, mean_fid - std_fid, mean_fid + std_fid, alpha=0.2)

# Add labels, legend, and title
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('Average FID over Epochs with STD Range')
plt.legend()
plt.grid()
plt.ylim([50, 180])
plt.tight_layout()

# Save the plot or show it

plt.savefig(f"/home/ymbahram/scratch/baselines_avg/a3ft/data10/augmentation_FID_KID.png")
plt.show()