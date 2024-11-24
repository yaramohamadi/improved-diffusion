import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['a3ft', 'finetune'] # 

for mode in modes: 

    for g in [0.05, 0.1, 1, 5]: # Fixed guidances I want to try
        for gamma in [10]:

            for dataset_size in [10]:
            
                file_path = f"/home/ymbahram/scratch/baselines/classifier-guidance/results_samesample/data{dataset_size}/FID_KID_intra.csv"

                print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                for epoch in ['000', '025', '050', '075', '100', '125', '150']:
                    
                    print("*"*20)
                    print(f"g {g} gamma {gamma} mode {mode} configuration {epoch} epoch")
                    print("*"*20)
                    sample_path = f"/home/ymbahram/scratch/clf_trg_results/results_samesample/g_p2_a3ft/data{dataset_size}/guided_sampling/{mode}/g{g}_gamma{gamma}/samples/samples_{epoch}.npz"
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

                    df_add = pd.DataFrame([results])

                    # Append
                    df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                    print(f"_______________________________{g} {mode} {epoch} has been written to {file_path}_______________________")

