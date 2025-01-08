import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file to inspect its structure
file_path = '/home/ymbahram/scratch/baselines/ddpm-pa/data10/FID_KID_lambdas.csv'
df = pd.read_csv(file_path)

df = df.groupby(['lambda_1', 'lambda_2', 'lambda_3'])['FID'].min().reset_index()


# Create a matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')

# Create and display the table
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Save the table as an image
plt.savefig('/home/ymbahram/scratch/baselines/ddpm-pa/data10/lambdas_dataframe.png', bbox_inches='tight', dpi=300)
plt.show()