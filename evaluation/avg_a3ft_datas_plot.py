import pandas as pd
import matplotlib.pyplot as plt

# Plot the data
plt.figure(figsize=(10, 6))

for dataset_size in [10, 500, 2503]:
    # Read the CSV file

    if dataset_size == 10:
        file_path = f"/home/ymbahram/scratch/baselines_avg/a3ft/data{dataset_size}/gamma0_FID_KID.csv"  # Replace with your CSV file path
    else:
        file_path = f"/home/ymbahram/scratch/baselines_avg/a3ft/data{dataset_size}/FID_KID.csv"  # Replace with your CSV file path
    df = pd.read_csv(file_path)

    # Group by p2_gamma and epoch, calculate mean and standard deviation of FID
    summary = df.groupby(["epoch"])["FID"].agg(["mean", "std"]).reset_index()



    epochs = summary["epoch"]
    mean = summary["mean"]
    std = summary["std"]

    # Compute upper and lower bounds
    upper_bound = mean + std
    lower_bound = mean - std

    # Plot the mean line
    plt.plot(epochs, mean, label=f"Dataset size = {dataset_size}", marker="o")

    # Add the fill_between for the error region
    plt.fill_between(epochs, lower_bound, upper_bound, alpha=0.2)


# Customize plot
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID over Epochs for Different dataset sizes (Averaged over Repetitions)")
plt.legend(title="Dataset sizes")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("/home/ymbahram/scratch/baselines_avg/a3ft/datasets_FID_KID.png")