import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
 
import copy
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
 
# ______________________ Sample related ________________________
how_many_samples= 10000 
batch_size=10
evaluate = True 

# IMAGENET IDDPM Configuration
image_size=64
num_channels=128
num_res_blocks=3
num_heads=4
num_heads_upsample=-1
attention_resolutions="16,8"
dropout=0.0
learn_sigma=True
diffusion_steps=4000
noise_schedule="cosine"
use_kl=False
 
# Other
sigma_small=False
class_cond=False
predict_xstart=False
rescale_timesteps=True
rescale_learned_sigmas=True
use_checkpoint=False # To do gradient checkpointing
use_scale_shift_norm=True
 
# Sampling and evaluating while training
timestep_respacing="ddim50"
use_ddim=True
sample = True # Doing sampling for a batch in training every time saving
image_size=image_size
evaluate = False # If you want to perform evaluation during training (Currently every 25 steps)


# Fixed noise vector
# noise_vector = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/pokemon_fixed_noise.npy'
noise_vector = '/export/livia/home/vision/Ymohammadi/util_files/pokemon_fixed_noise_10k.npy'


# Load the noise vector from the .npy file
noise_vector = th.tensor(np.load(noise_vector)).to('cuda')

# ____________________ Model ____________________

for repetition in range(3):

    for p2_gamma in [0]:

        # PATHS  
            load_model_paths = [
            f'/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/gamma0_repeat0/checkpoints//model000450.pt',
            f'/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/gamma0_repeat0/checkpoints//model000400.pt',
            f'/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/gamma0_repeat0/checkpoints//model000500.pt',
            ]

            save_samples_dirs= [
            f"/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/samples_10k/",
            f"/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/samples_10k/",
            f"/export/livia/home/vision/Ymohammadi/baselines_avg/a3ft/data10/samples_10k/"
            ]

            model = create_model(
                image_size = image_size,
                num_channels = num_channels,
                num_res_blocks = num_res_blocks,
                learn_sigma= learn_sigma,
                class_cond= class_cond,
                use_checkpoint= use_checkpoint,
                attention_resolutions=attention_resolutions,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
                dropout=dropout,
                time_aware=True,
            )

            pretrained_model = create_model(
                image_size = image_size,
                num_channels = num_channels,
                num_res_blocks = num_res_blocks,
                learn_sigma= learn_sigma,
                class_cond= class_cond,
                use_checkpoint= use_checkpoint,
                attention_resolutions=attention_resolutions,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
                dropout=dropout,
                time_aware=True, # TIME-AWARE
            )

            # ________________ Load Pretrained ____________

            checkpoint = th.load(load_model_paths[repetition])
            model.load_state_dict(checkpoint, strict = True) # TIMEAWARE: Because we are adding some new modules  
            pretrained_model.load_state_dict(checkpoint, strict = True) # TIMEAWARE: Because we are adding some new modules  

            model.to('cuda')
            pretrained_model.to('cuda')
            
            diffusion = create_gaussian_diffusion(
                steps=diffusion_steps,
                learn_sigma=learn_sigma,
                sigma_small=sigma_small,
                noise_schedule=noise_schedule,
                use_kl=use_kl,
                predict_xstart=predict_xstart,
                rescale_timesteps=rescale_timesteps,
                rescale_learned_sigmas=rescale_learned_sigmas,
                timestep_respacing=timestep_respacing,
                p2_gamma=p2_gamma, 
            )

            print(f"sampling {how_many_samples} images")
            sample_fn = (diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop)

            # Create folder
            im_path = os.path.join(save_samples_dirs[repetition], f'10k_samples_repeat{repetition}')
            os.makedirs(im_path, exist_ok=True)

            all_images = []
            for ind, _ in tqdm(enumerate(range(0, how_many_samples, batch_size))):

                # Fixing for same sample generation (Deterministic sampling is done by default in the code)
                if noise_vector != None:
                    initial_noise = noise_vector[ind]
                else: 
                    initial_noise = None

                sample = sample_fn(
                    model,
                    (batch_size, 3, image_size , image_size),
                    source_model=pretrained_model, # classifier-free guidance  guidance = th.tensor([guidance], device='cuda', dtype=th.float32) 
                    guidance=False, # classifier-free guidance
                    noise=initial_noise,
                    clip_denoised=True,
                    model_kwargs={}, # This is not needed, just class conditional stuff
                    progress=False
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous().cpu().numpy()

                if ind <5: # Save 5 batches as images to see the visualizations
                    for sidx, s in enumerate(sample):
                        plt.imsave(os.path.join(im_path, f'{sidx + ind*batch_size}.png'), s)
                all_images.extend(sample)

            all_images = all_images[: how_many_samples]
            
            sample_path = os.path.join(save_samples_dirs[repetition], f"10k_samples_repeat{repetition}")
            np.savez(sample_path, all_images)
            print("sampling complete")