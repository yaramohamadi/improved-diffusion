import sys
import pandas as pd
import numpy as np

from evaluation_util import runEvaluate

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'



for g in [
    'a5-0_8-1', 'a7_5-0_8-1']:
    
    data_list = []

    for iteration in np.arange(0, 201, 25):
        sample_path = f'/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/curved_schedule/{g}/samples/samples_{iteration}.npz'
        results = runEvaluate(ref_path, sample_path, verbose=True)
        data_list.append(results)

    df = pd.DataFrame(data_list)
    csv_file = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/curved_schedule/{g}/evaluation.csv"
    df.to_csv(csv_file, index=False)

    print(f"{g} has been written to {csv_file}")