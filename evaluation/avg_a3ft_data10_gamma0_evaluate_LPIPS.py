import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['a3ft', 'finetune'] 

for repetition in range(3): #

    for mode in modes: 
        for p2_gamma in [0]:  #  0.1, 0.5, 1

            for g, g_name in {0:'0'
                    }.items(): 

                for dataset_size in [10]:

                    file_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/gamma0_LPIPS.csv"

                    print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                    if mode == 'a3ft': 
                        epochs = [0, 100, 200, 300, 325, 350, 375, 400, 425, 450, 475, 500]
                    elif mode == 'finetune':
                        epochs = [0, 25, 50, 75, 100, 125]

                    for epoch in epochs:
                        
                        print("*"*20)
                        print(f"'repetition: ', {repetition}, {g_name} {mode} configuration {epoch} epoch")
                        print("*"*20)
                        sample_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/gamma{p2_gamma}_repeat{repetition}/samples/samples_{epoch}.npz"
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
                    
