import sys
import pandas as pd
import numpy as np

from evaluation import runEvaluate

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'

for g, g_name in {
    0.25: '0_25-1', 0.5: '0_5-1', 0.75: '0_75-1', 0.9: '0_9-1'
    }.items():
    
    data_list = []

    for iteration in np.arange(0, 201, 25):
        sample_path = f'/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/time_linear_guidance/{g_name}/samples/samples_{iteration}.npz'
        results = runEvaluate(ref_path, sample_path, verbose=True)
        data_list.append(results)

    df = pd.DataFrame(data_list)
    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/time_linear_guidance/{g_name}/evaluation.csv"
    df.to_csv(csv_file, index=False)

    print(f"{g} has been written to {csv_file}")