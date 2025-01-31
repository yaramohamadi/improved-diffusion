# Re-importing necessary libraries since the execution environment was reset
import pandas as pd
import matplotlib.pyplot as plt



import os
import yaml
import socket

# Load YAML configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    server_name = socket.gethostname()
    server_config = config["servers"].get(server_name, {})
    if not server_config:
        raise ValueError(f"No configuration found for server: {server_name}")
    common_config = config.get("common", {})
    return {**common_config, **server_config}

config = load_config()

# Extract variables from the configuration
base_path = config["base_path"]



# Re-load the CSV file
file_path = os.path.join(base_path, f'clf_results/clf_xs_xt/time_p2/FID_KID.csv')
save_path = os.path.join(base_path, f'clf_results/clf_xs_xt/time_p2/FID.png')
data = pd.read_csv(file_path)

# Ensure data is properly loaded
data.head()

# Plot FID over epoch for each unique t_gamma value
unique_t_gamma = data['t_gamma'].unique()
plt.figure(figsize=(10, 6))

for t_gamma in unique_t_gamma:
    subset = data[data['t_gamma'] == t_gamma]
    plt.plot(subset['epoch'], subset['FID'], label=f't_gamma={t_gamma}')

gt = [
    174.4147199,
    100.6159389,
    82.57906991,
    74.75163453,
    72.29262298,
    78.39891706,
    83.65326290
]

plt.plot(subset['epoch'], gt, label=f'fine-tuning')


plt.title('FID over Epoch for each t_gamma')
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.legend(title='t_gamma')
plt.grid(True)
plt.savefig(save_path)