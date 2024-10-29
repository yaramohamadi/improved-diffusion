import pandas as pd
import matplotlib.pyplot as plt

# File paths for the CSVs
file_paths = ['/home/ymbahram/scratch/clf_trg_results/results_samesample/data10_guidedsample/0/intra_LPIPS_evaluation.csv', 
              '/home/ymbahram/scratch/clf_trg_results/results_samesample/data10_guidedsample/0_05/intra_LPIPS_evaluation.csv', 
              '/home/ymbahram/scratch/clf_trg_results/results_samesample/data10_guidedsample/0_1/intra_LPIPS_evaluation.csv']


df_0 = pd.read_csv(file_paths[0])
df_0_05 = pd.read_csv(file_paths[1])
df_0_1 = pd.read_csv(file_paths[2])
df_sobersample = pd.read_csv('/home/ymbahram/scratch/clf_trg_results/results_samesample/data10/evaluation_all.csv')

# Extract the FID column
kid_0_sober = df_sobersample[df_sobersample['g'] == 0]['intra_LPIPS']
kid_0_05_sober = df_sobersample[df_sobersample['g'] == 0.05]['intra_LPIPS']
kid_0_1_sober = df_sobersample[df_sobersample['g'] == 0.1]['intra_LPIPS']

# Define the x values as 0, 25, 50, 75, ..., based on the number of rows
x_values = [i * 25 for i in range(9)]

initial_g = 0.05
# Plot the FID values from the three CSVs
plt.plot(x_values, kid_0_1_sober, label=f'training G = {initial_g} - sampling G = 0')
plt.plot(x_values, df_0_05[df_0_05['gsample'] == 0.05]['intra_LPIPS'], label=f'training G = {initial_g} - sampling G = 0.05')
plt.plot(x_values, df_0_05[df_0_05['gsample'] == 0.1]['intra_LPIPS'], label=f'training G = {initial_g} - sampling G = 0.1')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('intra_LPIPS')
plt.title(f'intra_LPIPS Comparison Across Different Sampling Guidances (Trained on {initial_g} fixed guidance)')

# Show legend
plt.legend()

plt.ylim([0, 2500])
# Show the plot
plt.savefig(f'/home/ymbahram/scratch/clf_trg_results/results_samesample/data10_guidedsample/intraLPIPS_g{initial_g}_gsample.png')
