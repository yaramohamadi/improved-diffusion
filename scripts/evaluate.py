import sys
import pandas as pd
import numpy as np

from evaluation_util import runEvaluate

ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'

data_list = []

for iteration in np.arange(0, 505, 25):
    sample_path = f'/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/samples/samples_{iteration}.npz'
    results = runEvaluate(ref_path, sample_path, verbose=True)
    data_list.append(results)

df = pd.DataFrame(data_list)
csv_file = "/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/evaluation.csv"
df.to_csv(csv_file, index=False)

print(f"Data has been written to {csv_file}")