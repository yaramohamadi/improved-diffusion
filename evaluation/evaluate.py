import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['finetune'] 

for mode in modes: 

    for g in [0.01, 0.05, 0.1]: 
            
        for gamma in [0, 0.1, 1, 10]:

            for dataset_size in [10]:
            
                file_path = f"/home/ymbahram/scratch/clf_trg_results/results_samesample/g_p2_a3ft/data{dataset_size}/{mode}/FID_KID.csv"
                #df = pd.read_csv(file_path)
                #first_epoch = df['epoch'].max()
                first_epoch = 125

                print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                for epoch in np.arange(first_epoch + 25, 151, 25):
                    
                    print("*"*20)
                    print(f"g {g} gamma {gamma} mode {mode} configuration {epoch} epoch")
                    print("*"*20)
                    sample_path = f"/home/ymbahram/scratch/clf_trg_results/results_samesample/g_p2_a3ft/data{dataset_size}/{mode}/g{g}_gamma{gamma}/samples/samples_{epoch}.npz"
                    results = evaluation.runEvaluate(ref_path, sample_path, 
                                        FID=True, 
                                        #IS=True, 
                                        #sFID=True, 
                                        #prec_recall=True, 
                                        KID=True, 
                                        # LPIPS=True, source_batch=source_batch, 
                                        intra_LPIPS=False, 
                                        # target_batch=target_path, 
                                        verbose=True)
                    
                    results['epoch'] = epoch
                    results['mode'] = mode
                    results['gamma'] = gamma

                    df_add = pd.DataFrame([results])

                    # Append
                    df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                    print(f"_______________________________{g} {mode} {epoch} has been written to {file_path}_______________________")
                
