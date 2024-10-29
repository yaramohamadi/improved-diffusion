import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors

data_list = []

for g, g_name in {0:'0',0.05:'0_05',0.1:'0_1'
        }.items(): 
    
    for p2_gamma in [0, 0.1, 0.3, 1, 3, 10]:

        if g == 0 and p2_gamma in [0.1, 0.3, 1, 3, 10]:
            continue

        for epoch in np.arange(0, 201, 25):
            
            if g == 0.1 and (epoch == 175 or epoch == 200):
                continue

            print("*"*20)
            print(f"{g_name} {p2_gamma} configuration {epoch} epoch")
            print("*"*20)
            sample_path = f'/home/ymbahram/scratch/clf_trg_results/results_samesample/time-step/data10_2/{g_name}_{p2_gamma}/samples/samples_{epoch}.npz'
            results = evaluation.runEvaluate(ref_path, sample_path, 
                                #FID=True, 
                                #IS=True, 
                                #sFID=True, 
                                #prec_recall=True, 
                                KID=True, 
                                # LPIPS=True, source_batch=source_batch, 
                                # intra_LPIPS=True, target_batch=target_path, 
                                verbose=True)

            results['epoch'] = epoch
            results['g'] = g
            results['p2_gamma'] = p2_gamma
            data_list.append(results)

            df = pd.DataFrame(data_list)
            csv_file = f"/home/ymbahram/scratch/clf_trg_results/results_samesample/time-step/data10_2/evaluation.csv"
            df.to_csv(csv_file, index=False)

            print(f"_______________________________{g_name} {p2_gamma} has been written to {csv_file}_______________________")
        
