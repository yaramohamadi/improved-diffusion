import torch as th
import numpy as np

# Set the seed for reproducibility
seed = 42
th.manual_seed(seed)
np.random.seed(seed)

# Parameters for noise
num_of_batches = 500
batch_size = 10
channels = 3
height = 64
width = 64
file_name = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/code_for_parse/pokemon_fixed_noise.npy'
device = 'cuda'

# Generate random noise vector
noise_vector = th.randn((num_of_batches, batch_size, channels, height, width)).to(device)

# Move noise vector to CPU for saving
noise_vector_cpu = noise_vector.cpu().numpy()

# Save noise vector as a .npy file
np.save(file_name, noise_vector_cpu)

print(f"Fixed noise vector of ({num_of_batches}, {batch_size}, {channels}, {height}, {width}) saved to {file_name}.")