import sys
import torch
del sys.modules["torch"] 
del torch

from evaluation import runEvaluate
ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'
sample_path = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/samples/samples_0.npz'
runEvaluate(ref_path, sample_path, #sqrtm_func=sqrtm, 
            verbose=True)
