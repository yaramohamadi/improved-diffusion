import pandas as pd

def analyze_fid(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure the CSV has the required columns
    required_columns = {"FID", "epoch", "gamma_distil"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The CSV must contain the columns: {required_columns}")

    # Get unique gamma_distil values
    gamma_values = data["gamma_distil"].unique()

    print("Analysis of FID for each gamma_distil:")
    print("=" * 50)

    for gamma in gamma_values:
        # Filter the data for the current gamma_distil value
        gamma_data = data[data["gamma_distil"] == gamma]

        # Calculate the average FID per epoch
        avg_fid_per_epoch = gamma_data.groupby("epoch")["FID"].mean()

        # Find the best average FID over an epoch
        best_avg_fid = avg_fid_per_epoch.min()
        best_epoch_for_avg_fid = avg_fid_per_epoch.idxmin()

        # Find the best single FID
        best_single_fid = gamma_data["FID"].min()
        best_single_fid_epoch = gamma_data.loc[gamma_data["FID"].idxmin(), "epoch"]

        # Print results
        print(f"Gamma_distil = {gamma}:")
        print(f"  Best Average FID: {best_avg_fid:.4f} (at epoch {best_epoch_for_avg_fid})")
        print(f"  Best Single FID: {best_single_fid:.4f} (at epoch {best_single_fid_epoch})")
        print("-" * 50)

# Example usage
csv_file = "/home/ymbahram/scratch/baselines_avg/SDFT/data10/p2gamma0_FID_KID.csv"  # Replace with your CSV file path
analyze_fid(csv_file)