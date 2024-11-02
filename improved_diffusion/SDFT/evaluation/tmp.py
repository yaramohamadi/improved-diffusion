import pandas as pd
import matplotlib.pyplot as plt



def smooth_data(series, window_size=25):
    """Applies a moving average smoothing."""
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

config = "gamma_aux3gamma_distil3"

def plot_columns_from_csv(file_path):
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ["mse", "L_distill", "L_aux"]
        if not all(col in df.columns for col in required_columns):
            print("Required columns are not present in the CSV.")
            return

        # Plotting
        plt.figure(figsize=(10, 6))
        x_values = df.index * 25

        for col in required_columns:
            plt.plot(x_values, smooth_data(df[col]), label=col)

        plt.title(config)
        plt.xlabel("epoch")
        plt.ylabel("Values (Smoothened)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/lambda_0.1/{config}.png")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (replace with your file path)
file_path = f"/home/ymbahram/scratch/baselines/SDFT/results_samesample/data10/lambda_0.1/{config}/trainlog.csv"
plot_columns_from_csv(file_path)