import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['finetune', 'att']

for mode in modes: 

    for g, g_name in {0:'0'
            }.items(): 

        for dataset_size in [10, 100, 700, 2503]:

            data_list = []

            print("__________________________ STARTING FROM FIRST EPOCH_____________________")

            for epoch in np.arange(0, 501, 25):
                
                print("*"*20)
                print(f"{g_name} {mode} configuration {epoch} epoch")
                print("*"*20)
                sample_path = f"/home/ymbahram/scratch/baselines/a3ft/results_samesample/data{dataset_size}/{mode}/samples/samples_{epoch}.npz"
                results = evaluation.runEvaluate(ref_path, sample_path, 
                                    FID=True, 
                                    #IS=True, 
                                    #sFID=True, 
                                    #prec_recall=True, 
                                    KID=True, 
                                    # LPIPS=True, source_batch=source_batch, 
                                    # intra_LPIPS=True, target_batch=target_path, 
                                    verbose=True)
                
                results['epoch'] = epoch
                results['mode'] = mode
                data_list.append(results)

                df = pd.DataFrame(data_list)
                csv_file = f"/home/ymbahram/scratch/baselines/a3ft/results_samesample/data{dataset_size}/{mode}/FID_KID.csv"
                df.to_csv(csv_file, index=False)

                print(f"_______________________________{g} {mode} {epoch} has been written to {csv_file}_______________________")