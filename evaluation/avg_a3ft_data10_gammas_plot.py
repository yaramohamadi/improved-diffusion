import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = "/home/ymbahram/scratch/baselines_avg/a3ft/data10/gammas_FID_KID.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Group by p2_gamma and epoch, calculate mean and standard deviation of FID
summary = df.groupby(["p2_gamma", "epoch"])["FID"].agg(["mean", "std"]).reset_index()

# Plot the data
plt.figure(figsize=(10, 6))

for gamma in summary["p2_gamma"].unique():
    gamma_data = summary[summary["p2_gamma"] == gamma]
    # Assuming gamma_data contains the data

    epochs = gamma_data["epoch"]
    mean = gamma_data["mean"]
    std = gamma_data["std"]

    # Compute upper and lower bounds
    upper_bound = mean + std
    lower_bound = mean - std

    # Plot the mean line
    plt.plot(epochs, mean, label=f"p2_gamma = {gamma}", marker="o")

    # Add the fill_between for the error region
    plt.fill_between(epochs, lower_bound, upper_bound, alpha=0.2)


# Customize plot
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID over Epochs for Different p2_gamma Values (Averaged over Repetitions)")
plt.legend(title="p2_gamma")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("/home/ymbahram/scratch/baselines_avg/a3ft/data10/gamma0_FID_KID.png")