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


import os
import pandas as pd
import re
import shutil


def clean_folders(base_dir):
    # Walk through all folders named 'repeat{number}'
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            # Match folders named 'repeat{number}'
            match = re.search(r'repeat(\d)', dir_name)
            if match:
                repetition_num = int(match.group(1))
                repeat_folder = os.path.join(root, dir_name)
                
                # Read FID_KID.csv in the current folder
                fid_kid_csv = os.path.join(root, 'FID_KID.csv')
                if not os.path.exists(fid_kid_csv):
                    print(f"FID_KID.csv not found in {root}")
                    continue
                
                df = pd.read_csv(fid_kid_csv)
                if 'FID' not in df.columns or 'epoch' not in df.columns or 'repetition' not in df.columns:
                    print(f"FID_KID.csv in {root} does not have required columns")
                    continue

                # Find the minimum FID for this repetition
                repetition_data = df[df['repetition'] == repetition_num]
                if repetition_data.empty:
                    print(f"No data for repetition {repetition_num} in {fid_kid_csv}")
                    continue
                
                min_fid_row = repetition_data.loc[repetition_data['FID'].idxmin()]
                target_epoch = str(int(min_fid_row['epoch']))
                print(f"Repetition {repetition_num}, Min FID: {min_fid_row['FID']}, Epoch: {target_epoch}")

                # Paths to checkpoints and samples
                checkpoints_dir = os.path.join(repeat_folder, 'checkpoints')
                samples_dir = os.path.join(repeat_folder, 'samples')

                # Remove files and folders not containing the target epoch
                for folder in [checkpoints_dir, samples_dir]:
                    if os.path.exists(folder):
                        for item in os.listdir(folder):
                            item_path = os.path.join(folder, item)
                            if target_epoch not in item:
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                    print(f"Deleted file: {item_path}")
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                                    print(f"Deleted folder: {item_path}")
                            else:
                                print(item_path)

# Specify the base directory where the folders are located

base_directory = os.path.join(base_path, 'baselines_avg/a3ft/data10/')
clean_folders(base_directory)