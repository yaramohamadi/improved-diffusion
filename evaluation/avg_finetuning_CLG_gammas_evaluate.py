import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors

for repetition in range(3):
    for g in [0.05,
                0.1,
                1,
                5,]: # Fixed guidances I want to try
        
        for gamma in [0.1, 0.5, 1, 10]:
            for dataset_size in [10]:

                file_path = f"/home/ymbahram/scratch/baselines_avg/classifier-guidance/data{dataset_size}/FID_KID.csv"

                print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                for epoch in range(0, 151, 25):
                    
                    print("*"*20)
                    print(f"g {g} gamma {gamma} configuration {epoch} epoch")
                    print("*"*20) 
                    sample_path = f"/home/ymbahram/scratch/baselines/classifier-guidance/data{dataset_size}/gamma{gamma}_g{g}/samples/samples_{epoch}.npz"
                    results = evaluation.runEvaluate(ref_path, sample_path, 
                                        FID=True, 
                                        #IS=True, 
                                        #sFID=True, 
                                        #prec_recall=True, 
                                        KID=True, 
                                        # LPIPS=True, source_batch=source_batch, 
                                        # intra_LPIPS=True, 
                                        target_batch=target_path, 
                                        verbose=True)
                    
                    results['epoch'] = epoch
                    results['gamma'] = gamma
                    results['g'] = g
                    results['repetition'] = repetition

                    df_add = pd.DataFrame([results])

                    # Append
                    df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                    print(f"_______________________________{g} {epoch} has been written to {file_path}_______________________")