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
    
for repetition in range(3):
    for g in [1]: # Fixed guidances I want to try
        for gamma in [0]:
            for dataset_size in [10]:

                    file_path = f"/export/livia/home/vision/Ymohammadi/baselines_avg/CLG/data10/samples_10k/evaluate_all.csv"
                    
                    print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                    sample_path = f"/export/livia/home/vision/Ymohammadi/baselines_avg/CLG/data10/samples_10k/10k_samples_repeat{repetition}.npz"
                    results = evaluation.runEvaluate(ref_path, sample_path, 
                                        FID=True, 
                                        IS=True, 
                                        # sFID=True, 
                                        prec_recall=True, 
                                        KID=True, 
                                        LPIPS=True, source_batch=source_batch, 
                                        intra_LPIPS=True, target_batch=target_path, 
                                        #verbose=True
                                        )
                    
                    results['gamma'] = gamma
                    results['g'] = g
                    results['repetition'] = repetition

                    df_add = pd.DataFrame([results])

                    # Append
                    df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                    print(f"_______________________________has been written to {file_path}_______________________")
            
