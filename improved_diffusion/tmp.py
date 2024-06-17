import evaluation
ref_path = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'
sample_path = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/samples/samples_0.npz'
evaluation.runEvaluate(ref_path, sample_path, verbose=True)
