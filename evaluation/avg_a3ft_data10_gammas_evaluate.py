
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)


# ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
# target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
# source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
ref_path = '/export/livia/home/vision/Ymohammadi/datasets/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/export/livia/home/vision/Ymohammadi/datasets/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/export/livia/home/vision/Ymohammadi/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
modes = ['a3ft'] 

for repetition in range(0,2,1):

    for mode in modes: 
        for p2_gamma in [0]:  # 0 # 0.1, 0.5, 1

            for g, g_name in {0:'0'
                    }.items(): 

                for dataset_size in [10]:

                    file_path = f"/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/gammas_FID_KID.csv"

                    print("__________________________ STARTING FROM FIRST EPOCH_____________________")

                    for epoch in range(0, 1501, 50):
                        
                        print("*"*20)
                        print(f"'repetition: ', {repetition}, {g_name} {mode} configuration {epoch} epoch")
                        print("*"*20)
                        sample_path = f"/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/gamma{p2_gamma}_repeat{repetition}/samples/samples_{epoch}.npz"
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

                        df_add = pd.DataFrame([results])

                        # Append
                        df_add.to_csv(file_path, mode="a", index=False, header=not pd.io.common.file_exists(file_path))

                        print(f"_______________________________repetition {repetition} config {g} {mode} {epoch} has been written to {file_path}_______________________")
                    
