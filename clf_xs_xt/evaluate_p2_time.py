
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

t_gammas = [
    50,
    10,
    2,
    0.5,
    0.1,
    0.01,
]

for repetition in range(1):

        for t_gamma in t_gammas: # Fixed guidances I want to try
            for gamma in [0]:

                for dataset_size in [10]:
                                 
                    file_path = os.path.join(base_path, f'clf_results/clf_xs_xt/time_p2/FID_KID.csv')


                    for epoch in [0, 25, 50, 75, 100, 125, 150]:
                        
                        print("*"*20)
                        print(f"t_gamma {t_gamma} gamma {gamma} configuration {epoch} epoch")
                        print("*"*20)

                        sample_path = os.path.join(base_path, f'clf_results/clf_xs_xt/time_p2/t_gamma_reverse{t_gamma}/samples/samples_{epoch}.npz')
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
                        results['t_gamma'] = t_gamma
                        results['repetition'] = repetition

                        df_add = pd.DataFrame([results])

                        # Append
                        df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                        print(f"_______________________________ g_k {t_gamma} repetition {repetition} epoch {epoch} has been written to {file_path} _______________________")

