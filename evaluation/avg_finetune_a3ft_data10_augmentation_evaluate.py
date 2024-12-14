import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
modes = ['a3ft', 'finetune'] 

for repetition in range(3):

    for mode in modes: 
        for p2_gamma in [0]:

            for g, g_name in {0:'0'
                    }.items(): 

                for dataset_size in [10]:

                        file_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/augmentation_FID_KID.csv"

                        print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                        

                        
                        for augmentation in ['Flip', 'Much', 'None']:
                            if mode == 'a3ft':
                                epochs = [0, 100, 200, 300, 325, 350, 375, 400, 425, 450, 475, 500]
                            else:
                                if augmentation == 'Flip':
                                    epochs = [0, 25, 50, 75, 100, 125]
                                else:
                                    epochs = [0, 25, 50, 75, 100, 125, 150, 175, 200]

                            for epoch in epochs:

                                if augmentation in ['Much', 'None']:
                                    sample_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/{p2_gamma}_repeat{repetition}_augmentation{augmentation}/samples/samples_{epoch}.npz"
                                    if repetition == 2:
                                        continue
                                else:
                                    if mode == 'finetune':
                                        sample_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/{p2_gamma}_repeat{repetition}/samples/samples_{epoch}.npz"
                                    else:
                                        sample_path = f"/home/ymbahram/scratch/baselines_avg/{mode}/data{dataset_size}/gamma{p2_gamma}_repeat{repetition}/samples/samples_{epoch}.npz"

                            
                                print("*"*20)
                                print(f"'repetition: ', {repetition}, {g_name} {mode} configuration {epoch} epoch")
                                print("*"*20)  
                                
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
                                results['repetition'] = repetition
                                results['p2_gamma'] = p2_gamma
                                results['augmentation'] = augmentation

                                df_add = pd.DataFrame([results])

                                # Append
                                df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                                print(f"_______________________________repetition {repetition} config {g} {mode} {epoch} has been written to {file_path}_______________________")
                            
