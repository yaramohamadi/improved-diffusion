import pandas as pd
import matplotlib.pyplot as plt


# Define the lambda_distils and gamma_distils we're interested in plotting
lambda_distils_to_plot = [#0.1, 0.3, 1
    0.001, 0.005, 0.1]
gamma_distils_to_plot = [0 , 0.1, 1
    #10, 30, 100
    ]

# Define the path for the baseline file (lambda_distil = 0)
baseline_path = '/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/distil_ablate/lambda_distil_only_0/FID_KID.csv'

# Load baseline data
baseline_data = pd.read_csv(baseline_path)

# Placeholder dictionary to store data for each lambda_distil
data_by_lambda = {ld: [] for ld in lambda_distils_to_plot}

# Loop through the lambda_distils and gamma_distils to load and plot
for lambda_distil in lambda_distils_to_plot:
    for gamma_distil in gamma_distils_to_plot:
        sample_path = f'/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/distil_ablate/lambda_distil_only_{lambda_distil}/FID_KID_newHyperparameters.csv'
        try:
            # Load data
            data = pd.read_csv(sample_path)
            # Filter for the specific lambda_distil and gamma_distil values
            filtered_data = data[data['gamma_distils'] == gamma_distil]
            data_by_lambda[lambda_distil].append((gamma_distil, filtered_data))
        except FileNotFoundError:
            print(f"File not found for lambda_distil {lambda_distil} and gamma_distil {gamma_distil}")

# Plotting
fig, axes = plt.subplots(1, len(lambda_distils_to_plot), figsize=(18, 6), sharey=True)

for idx, lambda_distil in enumerate(lambda_distils_to_plot):
    ax = axes[idx]
    # Plot baseline
    ax.plot(baseline_data['epoch'], baseline_data['KID'], label="Baseline (λ=0)", linestyle="--", color="black")
    
    # Plot each gamma_distil line for the current lambda_distil
    for gamma_distil, gamma_data in data_by_lambda[lambda_distil]:
        ax.plot(gamma_data['epoch'], gamma_data['KID'], label=f"γ={gamma_distil}")

    ax.set_title(f"λ={lambda_distil}")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)

axes[0].set_ylabel("KID")
plt.suptitle("KID vs Epoch for Different λ and γ Values")
plt.savefig('/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/distil_ablate/KID_newHyperparameters.png')