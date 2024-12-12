import pandas as pd
import matplotlib.pyplot as plt

def plot_fid_over_epochs(dataset_sizes):

    # Plotting
    plt.figure(figsize=(10, 6))

    for dataset_size in dataset_sizes:

        print(dataset_size)
        csv_file = f"/home/ymbahram/scratch/baselines_avg/SDFT/data{dataset_size}/p2gamma0_FID_KID.csv"  # Replace with your CSV file path

        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Ensure the CSV has the required columns
        required_columns = {"FID", "epoch", "gamma_distil"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"The CSV must contain the columns: {required_columns}")

        # Filter the data for the two specific dataset_size values
        data1 = data

        # Group by epoch and calculate mean and standard deviation for each gamma_distil
        stats1 = data1.groupby("epoch")["FID"].agg(["mean", "std"]).reset_index()

        print(stats1)

        # Plot the first dataset_size line
        plt.plot(stats1["epoch"], stats1["mean"], label=f"dataset_size = {dataset_size}")
        plt.fill_between(stats1["epoch"], 
                        stats1["mean"] - stats1["std"], 
                        stats1["mean"] + stats1["std"],  alpha=0.2)

    # Customize the plot
    plt.title("FID Over Epochs for Different Dataset sizes", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("FID", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("/home/ymbahram/scratch/baselines_avg/SDFT/FID_KID.png")

# Example usage
plot_fid_over_epochs(dataset_sizes= [10, 500, 2503])
