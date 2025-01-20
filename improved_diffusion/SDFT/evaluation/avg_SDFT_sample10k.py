import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/export/livia/home/vision/Ymohammadi/datasets/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/export/livia/home/vision/Ymohammadi/datasets/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/export/livia/home/vision/Ymohammadi/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    
for repetition in range(1, 3):

    for p2_gamma in [0]: # TODO Quoi??

        for lambda_1 in [0.1]:
            
            for lambda_2 in [0.5]:
                
                for lambda_3 in [0.08]:

                    for dataset_size in [10]:


                        file_path = f"/export/livia/home/vision/Ymohammadi/baselines_avg/ddpm-pa/samples_10k/evaluate_all.csv"
                        
                        print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                        sample_path = f"/export/livia/home/vision/Ymohammadi/baselines_avg/ddpm-pa/samples_10k/10k_samples_repeat{repetition}.npz"
                        results = evaluation.runEvaluate(ref_path, sample_path, 
                                            FID=True, 
                                            IS=True, 
                                            sFID=True, 
                                            prec_recall=True, 
                                            KID=True, 
                                            LPIPS=True, source_batch=source_batch, 
                                            intra_LPIPS=True, target_batch=target_path, 
                                            verbose=True
                                            )
                        
                        results['gamma'] = p2_gamma
                        results['lambda_1'] = lambda_1
                        results['lambda_2'] = lambda_2
                        results['lambda_3'] = lambda_3
                        results['repetition'] = repetition

                        df_add = pd.DataFrame([results])

                        # Append
                        df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                        print(f"_______________________________has been written to {file_path}_______________________")
                
