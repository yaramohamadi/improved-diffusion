import sys
import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset

    
data_list = []

for g, g_name in {0:'0', 0.1:'0_1', 0.05: '0_05', 0.2:'0_2'
        }.items(): 
    
    for epoch in np.arange(0, 201, 25):
        
        print("*"*20)
        print(f"{g_name} configuration {epoch} epoch")
        print("*"*20)
        sample_path = f'/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/{g_name}/samples/samples_{epoch}.npz'
        results = evaluation.runEvaluate(ref_path, sample_path, verbose=True)
        evaluation.runEvaluate(ref_path, sample_path, 
                               #FID=True, 
                               #IS=True, 
                               #sFID=True, 
                               #prec_recall=True, 
                               KID=True, 
                               #LPIPS=False, source_batch=None, 
                               intra_LPIPS=True, target_batch=target_path, 
                               verbose=True)

        results['epoch'] = epoch
        data_list.append(results)

    df = pd.DataFrame(data_list)
    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/evaluation2.csv"
    df.to_csv(csv_file, index=False)

    print(f"{g} has been written to {csv_file}")
    
    