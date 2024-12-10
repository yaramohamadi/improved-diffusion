import pandas as pd
import matplotlib.pyplot as plt

def plot_fid_over_epochs(csv_file, gamma_distil_1, gamma_distil_2):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure the CSV has the required columns
    required_columns = {"FID", "epoch", "gamma_distil"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The CSV must contain the columns: {required_columns}")

    # Filter the data for the two specific gamma_distil values
    data1 = data[data["gamma_distil"] == gamma_distil_1]
    data2 = data[data["gamma_distil"] == gamma_distil_2]

    # Group by epoch and calculate mean and standard deviation for each gamma_distil
    stats1 = data1.groupby("epoch")["FID"].agg(["mean", "std"]).reset_index()
    stats2 = data2.groupby("epoch")["FID"].agg(["mean", "std"]).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the first gamma_distil line
    plt.plot(stats1["epoch"], stats1["mean"], label=f"Gamma_distil = {gamma_distil_1}", color="blue")
    plt.fill_between(stats1["epoch"], 
                     stats1["mean"] - stats1["std"], 
                     stats1["mean"] + stats1["std"], 
                     color="blue", alpha=0.2)

    # Plot the second gamma_distil line
    plt.plot(stats2["epoch"], stats2["mean"], label=f"Gamma_distil = {gamma_distil_2}", color="orange")
    plt.fill_between(stats2["epoch"], 
                     stats2["mean"] - stats2["std"], 
                     stats2["mean"] + stats2["std"], 
                     color="orange", alpha=0.2)

    # Customize the plot
    plt.title("FID Over Epochs for Different Gamma_distil Values", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("FID", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig("/home/ymbahram/scratch/baselines_avg/SDFT/data10/p2gamma0_FID_KID.png")

# Example usage
csv_file = "/home/ymbahram/scratch/baselines_avg/SDFT/data10/p2gamma0_FID_KID.csv"  # Replace with your CSV file path
plot_fid_over_epochs(csv_file, gamma_distil_1=0.1, gamma_distil_2=1)