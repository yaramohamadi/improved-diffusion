import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['finetune'] # 'a3ft'

for mode in modes: 

    for g in [0.1, 1]: # Fixed guidances I want to try
        for gamma in [0, 0.1, 1]:

            for dataset_size in [10]:
                                
                file_path = f"/home/ymbahram/scratch/baselines_avg/classifier-free/data{dataset_size}/{mode}/FID_KID.csv"

                for epoch in [0, 25, 50, 75, 100, 125, 150]:
                    
                    print("*"*20)
                    print(f"g {g} gamma {gamma} mode {mode} configuration {epoch} epoch")
                    print("*"*20)
                    sample_path = f"/home/ymbahram/scratch/baselines_avg/classifier-free/data{dataset_size}/{mode}/g{g}_gamma{gamma}_repetition{repetition}/samples/samples_{epoch}.npz"
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
                    results['mode'] = mode
                    results['gamma'] = gamma
                    results['g'] = g
                    results['repetition'] = repetition

                    df_add = pd.DataFrame([results])

                    # Append
                    df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                    print(f"_______________________________ g {g} gamma {gamma} repetition {repetition} {mode} {epoch} has been written to {file_path} _______________________")

