import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/datasets/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors

lambda_auxs = [0.001] 
lambda_distils = [0.001]
# SDFT: Output from auxiliary input drastically collapses in smaller timesteps therefore larger gamma (Less influence in smaller timesteps)
gamma_auxs = [
    0.1]
gamma_distils = [0.1]

mode = 'SDFT'
for repetition in range(3):

    for lambda_distil, lambda_aux in zip(lambda_distils, lambda_auxs): # SDFT: We assume that these two hyperparameters should be the same, just like in the paper
            for gamma_aux, gamma_distil in zip(gamma_auxs, gamma_distils):

                for p2_gamma in [0]:

                    for g, g_name in {0:'0'
                            }.items(): 

                        for dataset_size in [10, 500, 2503]:

                            file_path = f"/home/ymbahram/scratch/baselines_avg/SDFT/dataset_LPIPS.csv"

                            print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                            if dataset_size == 10:
                                epochs = [100]# [175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
                            elif dataset_size == 500:
                                epochs = [775]
                            elif dataset_size == 2503:
                                epochs = [575]
                            for epoch in epochs:
                                
                                print("*"*20)
                                print(f"'repetition: ', {repetition}, {g_name} lambdas{lambda_distil} gammas{gamma_aux} configuration {epoch} epoch")
                                print("*"*20)
                                sample_path = f"/home/ymbahram/scratch/baselines_avg/SDFT/data{dataset_size}/p2_gamma{p2_gamma}_repeat{repetition}/lambdas{lambda_distil}_gammas{gamma_distil}/samples/samples_{epoch}.npz"
                                results = evaluation.runEvaluate(ref_path, sample_path, 
                                                    #FID=True, 
                                                    #IS=True, 
                                                    #sFID=True, 
                                                    #prec_recall=True, 
                                                    #KID=True, 
                                                    # LPIPS=True, source_batch=source_batch, 
                                                    # intra_LPIPS=True, target_batch=target_path, 
                                                    verbose=True)
                                
                                results['epoch'] = epoch
                                results['lambda_distil'] = lambda_distil
                                results['lambda_aux'] = lambda_aux
                                results['gamma_distil'] = gamma_distil
                                results['gamma_aux'] = gamma_aux
                                results['repetition'] = repetition
                                results['p2_gamma'] = p2_gamma

                                df_add = pd.DataFrame([results])

                                # Append
                                df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                                print(f"_______________________________repetition {repetition} config {g} lambdas{lambda_distil} gammas{gamma_aux} {epoch} has been written to {file_path}_______________________")
                            
