import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
lambda_auxs = [0] #0.1 , 0.3, 1]
lambda_distils = [0.001, 0.005, 0.1]
# SDFT: Output from auxiliary input drastically collapses in smaller timesteps therefore larger gamma (Less influence in smaller timesteps)
gamma_auxs = [#0, 0.1, 0.6, 3
    9999]
gamma_distils = [0, 0.1, 1]

for lambda_distil in lambda_distils: # SDFT: We assume that these two hyperparameters should be the same, just like in the paper

    data_list = []

    for gamma_aux in gamma_auxs:
        for gamma_distil in gamma_distils:

            for dataset_size in [10]:#, 100, 700, 2503]:

                for gsample, gsample_name in {0.0:'0'
                    }.items(): 

                    for epoch in np.arange(0, 151, 25):
                        
                        print("*"*20)
                        print(f"lambda_distil: {lambda_distil}, gamma_aux: {gamma_aux}, gamma_distil: {gamma_distil} epoch {epoch} ")
                        print("*"*20)

                        sample_path = f'/home/ymbahram/scratch/baselines/SDFT/results_samesample/data{dataset_size}/distil_ablate/lambda_distil_only_{lambda_distil}/gamma_distil{gamma_distil}/samples/samples_{epoch}.npz'
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
                        results['lambda_distil'] = lambda_distil
                        results['gamma_distils'] = gamma_distil
                        data_list.append(results)

                        df = pd.DataFrame(data_list)
                        csv_file = f"/home/ymbahram/scratch/baselines/SDFT/results_samesample/data{dataset_size}/distil_ablate/lambda_distil_only_{lambda_distil}/FID_KID_newHyperparameters.csv"
                        df.to_csv(csv_file, index=False)

    print(f"___________________________File Written____/lambda_distil_only_{lambda_distil}/FIDKID_evaluation.csv_______________________")
                
