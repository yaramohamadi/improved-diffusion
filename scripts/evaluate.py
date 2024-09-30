import sys
import pandas as pd
import numpy as np

from evaluation import runEvaluate

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'


    
data_list = []

for g, g_name in {0:'0', 0.1:'0_1', 0.05: '0_05', 0.2:'0_2'
        }.items(): 
    
    for epoch in np.arange(0, 201, 25):
        sample_path = f'/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/{g_name}/samples/samples_{epoch}.npz'
        results = runEvaluate(ref_path, sample_path, verbose=True)
        results['epoch'] = epoch
        data_list.append(results)

    df = pd.DataFrame(data_list)
    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data10/evaluation.csv"
    df.to_csv(csv_file, index=False)

    print(f"{g} has been written to {csv_file}")