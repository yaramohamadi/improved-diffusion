
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




import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)


ref_path = os.path.join(base_path, f'datasets/pokemon/pokemon_64x64.npz')
target_path = os.path.join(base_path, f'datasets/pokemon/pokemon_10.npz')
source_batch = os.path.join(base_path, f'util_files/imagenet_pretrained.npz')

ks = [
    1000,
    250,
    100,
    50,
    25,
    10,
    1
]

for repetition in range(1):

        for k in ks: # Fixed guidances I want to try
            for gamma in [0]:

                for dataset_size in [10]:
                                 
                    file_path = os.path.join(base_path, f'clf_results/clf_xs_xt/sigmoid/FID_KID.csv')


                    for epoch in [0, 25, 50, 75, 100, 125, 150]:
                        
                        print("*"*20)
                        print(f"g_k {k} gamma {gamma} configuration {epoch} epoch")
                        print("*"*20)

                        sample_path = os.path.join(base_path, f'clf_results/clf_xs_xt/sigmoid/g_k{k}/samples/samples_{epoch}.npz')
                        results = evaluation.runEvaluate(ref_path, sample_path, 
                                            FID=True, 
                                            #IS=True, 
                                            #sFID=True, 
                                            #prec_recall=True, 
                                            KID=True, 
                                            # LPIPS=True, source_batch=source_batch, 
                                            # intra_LPIPS=True, 
                                            # target_batch=target_path, 
                                            verbose=True)
                        
                        results['epoch'] = epoch
                        results['g_k'] = k
                        results['repetition'] = repetition

                        df_add = pd.DataFrame([results])

                        # Append
                        df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                        print(f"_______________________________ g_k {k} repetition {repetition} epoch {epoch} has been written to {file_path} _______________________")

