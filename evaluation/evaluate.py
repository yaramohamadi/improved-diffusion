import pandas as pd
import numpy as np
import importlib

import evaluation 
importlib.reload(evaluation)

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz' # The target full dataset
target_path = '/home/ymbahram/scratch/pokemon/pokemon_10.npz' # The target 10-shot dataset
source_batch = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet_pretrained.npz' # Source samples from pre-fixed noise vectors
    

for g, g_name in {0:'0', 0.05: '0_05', 0.1:'0_1',  # , 0.2:'0_2'
        }.items(): 
    
    data_list = []

    for gsample, gsample_name in {0.05: '0_05' , 0.1:'0_1',  # , 0.2:'0_2'
        }.items(): 

        for epoch, ep_name in zip(np.arange(0, 201, 25), ['000',  '025', '050', '075', '100', '125', '150', '175', '200']):
            
            print("*"*20)
            print(f"{g_name} {gsample_name} configuration {epoch} epoch")
            print("*"*20)
            sample_path = f'/home/ymbahram/scratch/clf_trg_results/results_samesample/data10_guidedsample/{g_name}/samples_{gsample_name}/samples_{ep_name}.npz'
            results = evaluation.runEvaluate(ref_path, sample_path, 
                                #FID=True, 
                                #IS=True, 
                                #sFID=True, 
                                #prec_recall=True, 
                                # KID=True, 
                                # LPIPS=True, source_batch=source_batch, 
                                intra_LPIPS=True, target_batch=target_path, 
                                verbose=True)

            results['epoch'] = epoch
            results['gsample'] = gsample
            data_list.append(results)

            df = pd.DataFrame(data_list)
            csv_file = f"/home/ymbahram/scratch/clf_trg_results/results_samesample/data10_guidedsample/{g_name}/intra_LPIPS_evaluation.csv"
            df.to_csv(csv_file, index=False)

            print(f"_______________________________{g} {gsample} {epoch} has been written to {csv_file}_______________________")
        
