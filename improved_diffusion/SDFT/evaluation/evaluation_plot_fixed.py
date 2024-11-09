import pandas as pd
import matplotlib.pyplot as plt


# Define the lambda_distils and gamma_distils we're interested in plotting
lambda_auxs_to_plot = [#0.1, 0.3, 1
    0.001, 0.005, 0.01, 0.1]
gamma_auxs_to_plot = [0 , 0.1, 1
    #10, 30, 100
    ]

metric='FID'

# Define the path for the baseline file (lambda_distil = 0)
baseline_path = '/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/aux_ablate/lambda_aux_only_0/FID_KID.csv'

# Load baseline data
baseline_data = pd.read_csv(baseline_path)

# Placeholder dictionary to store data for each lambda_distil
data_by_lambda = {ld: [] for ld in lambda_auxs_to_plot}

# Loop through the lambda_distils and gamma_distils to load and plot
for lambda_aux in lambda_auxs_to_plot:
    for gamma_aux in gamma_auxs_to_plot:
        sample_path = f'/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/aux_ablate/lambda_aux_only_{lambda_aux}/FID_KID_newHyperparameters.csv'
        try:
            # Load data
            data = pd.read_csv(sample_path)
            # Filter for the specific lambda_distil and gamma_distil values
            filtered_data = data[data['gamma_aux'] == gamma_aux]
            data_by_lambda[lambda_aux].append((gamma_aux, filtered_data))
        except FileNotFoundError:
            print(f"File not found for lambda_aux {lambda_aux} and gamma_aux {gamma_aux}")

# Plotting
fig, axes = plt.subplots(1, len(lambda_auxs_to_plot), figsize=(18, 6), sharey=True)

for idx, lambda_aux in enumerate(lambda_auxs_to_plot):
    ax = axes[idx]
    # Plot baseline
    ax.plot(baseline_data['epoch'], baseline_data[metric], label="Baseline (λ=0)", linestyle="--", color="black")
    
    # Plot each gamma_distil line for the current lambda_distil
    for gamma_aux, gamma_data in data_by_lambda[lambda_aux]:
        ax.plot(gamma_data['epoch'], gamma_data[metric], label=f"γ={gamma_aux}")

    ax.set_title(f"λ={lambda_aux}")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)

axes[0].set_ylabel(metric)
plt.suptitle(f"{metric} vs Epoch for Different λ and γ Values")
plt.savefig(f'/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/aux_ablate/{metric}_newHyperparameters.png')