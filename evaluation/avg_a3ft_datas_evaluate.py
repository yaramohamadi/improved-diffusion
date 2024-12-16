import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['a3ft'] 

for repetition in range(3): # 3

    for mode in modes: 
        for p2_gamma in [0]:  # 0

            for g, g_name in {0:'0'
                    }.items(): 

                for dataset_size in [10, 500, 2503]:

                    file_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/dataset_LPIPS.csv"

                    print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                    if dataset_size == 10:
                        epochs = [400]
                    elif dataset_size == 500:
                        epochs = [700]
                    elif dataset_size == 2503:
                        epochs = [600]
                    for epoch in epochs: # [0, 200, 400, 450, 500, 550, 600, 650, 700, 750, 800]:
                        
                        print("*"*20)
                        print(f"'repetition: ', {repetition}, {g_name} {mode} configuration {epoch} epoch")
                        print("*"*20)

                        if dataset_size == 10:
                            sample_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/gamma{p2_gamma}_repeat{repetition}/samples/samples_{epoch}.npz"
                        else:
                            sample_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/{p2_gamma}_repeat{repetition}/samples/samples_{epoch}.npz"
                        results = evaluation.runEvaluate(ref_path, sample_path, 
                                            # FID=True, 
                                            #IS=True, 
                                            #sFID=True, 
                                            #prec_recall=True, 
                                            # KID=True, 
                                            LPIPS=True, source_batch=source_batch, 
                                            # intra_LPIPS=True, target_batch=target_path, 
                                            verbose=True)
                        
                        results['epoch'] = epoch
                        results['mode'] = mode
                        results['repetition'] = repetition
                        results['p2_gamma'] = p2_gamma

                        df_add = pd.DataFrame([results])

                        # Append
                        df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                        print(f"_______________________________repetition {repetition} config {g} {mode} {epoch} has been written to {file_path}_______________________")
                    
