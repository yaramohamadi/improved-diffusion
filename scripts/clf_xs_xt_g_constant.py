

import copy
import numpy as np

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop


import os
import yaml
import socket

# Load YAML configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    server_name = socket.gethostname()
    server_config = config["servers"].get(server_name, {})
    if not server_config:
        raise ValueError(f"No configuration found for server: {server_name}")
    common_config = config.get("common", {})
    return {**common_config, **server_config}

config = load_config()

# Extract variables from the configuration
base_path = config["base_path"]



# Training  
epochs = 301 
batch_size=10
schedule_sampler="uniform" 
lr=1e-4
weight_decay=0.0
lr_anneal_steps=0
microbatch=-1  
ema_rate="0.9999" 
log_interval=10
save_interval=25
use_fp16=False
fp16_scale_growth=1e-3

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
sample = True, # Doing sampling for a batch in training every time saving
how_many_samples= 1 # 2500
image_size=image_size
evaluate = False # If you want to save npz to perform evaluation later (FID and stuff)

# PATHS   
# Load pretrained model from here 
load_model_path= os.path.join(base_path, "util_files/imagenet64_uncond_100M_1500K.pt")
# If you are resuming a previously aborted training, include the path to the checkpoint here
resume_checkpoint= ""
# Only need this if we are evaluating FID and stuff while training
ref_dataset_npz = os.path.join(base_path, 'datasets/pokemon/pokemon_64x64.npz')
# Fixed noise vector
noise_vector = os.path.join(base_path, 'util_files/pokemon_fixed_noise.npy')

# Load the noise vector from the .npy file
noise_vector = th.tensor(np.load(noise_vector)).to('cuda')


# ____________________ Model ____________________


for repetition in range(1):

    for p2_gamma in [0]: 

            for dataset_size in [10]:

                # The dataset you want to finetune on
                data_dir = os.path.join(base_path, f'datasets/pokemon/pokemon{dataset_size}/')
                source_data_dir = os.path.join(base_path, f'datasets/imagenet_samples5000/')

                source_data = load_data(
                    data_dir=source_data_dir,
                    batch_size=batch_size,
                    image_size=image_size,
                    class_cond=False,
                )

                data = load_data(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    image_size=image_size,
                    class_cond=False,
                )

                for g in [0, 0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:  # 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1 0.1    0.1, 

                    for gamma in [0]:

                        print("*"*20)
                        print(f"guidance is {g} gamma is {gamma}")
                        print("*"*20)

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
                            time_aware=False,
                        )

                        # ________________ Load Pretrained ____________

                        checkpoint = th.load(load_model_path)
                        model.load_state_dict(checkpoint, strict = True) # TIMEAWARE: Because we are adding some new modules  

                        model.to('cuda')
                        
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
                            p2_gamma=gamma, 
                        )

                        # ________________classifier-free guidance (DONT NEED TODO removes)_______________

                        # Fixed
                        guidance_scale = np.array([g for _ in range(epochs)]) # Fixed Line

                        # Where to log the training loss (File does not have to exist)
                        loss_logger = os.path.join(base_path, f"clf_results/clf_xs_xt/constant_variance/g{g}/trainlog.csv")
                        # If evaluation is true during training, where to save the FID stuff
                        eval_logger = os.path.join(base_path, f"clf_results/clf_xs_xt/constant_variance/g{g}/evallog.csv")
                        # Directory to save checkpoints in
                        checkpoint_dir = os.path.join(base_path, f"clf_results/clf_xs_xt/constant_variance/g{g}/tmpcheckpoints/")
                        # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
                        save_samples_dir= os.path.join(base_path, f"clf_results/clf_xs_xt/constant_variance/g{g}/samples/")

                        # ________________ Train _________________ 

                        schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

                        TrainLoop(
                            model=model,
                            diffusion=diffusion,
                            source_data=source_data,
                            data=data,
                            batch_size=batch_size,
                            microbatch=microbatch,
                            lr=lr,
                            ema_rate=ema_rate,
                            log_interval=log_interval,
                            save_interval=save_interval,
                            resume_checkpoint=resume_checkpoint,
                            use_fp16=use_fp16,
                            fp16_scale_growth=fp16_scale_growth,
                            schedule_sampler=schedule_sampler,
                            weight_decay=weight_decay,
                            lr_anneal_steps=lr_anneal_steps,
                            # next 2 For logging
                            loss_logger=loss_logger,
                            eval_logger=eval_logger,
                            checkpoint_dir = checkpoint_dir,
                            # next 4 For sampling
                            sample = True, # Doing sampling for a batch in training every time saving
                            use_ddim=use_ddim,
                            save_samples_dir=save_samples_dir,
                            how_many_samples=how_many_samples,
                            image_size=image_size,
                            # For classifier-free guidanace (We dont need these, TODO delete later)
                            guidance_scale=guidance_scale,
                            clf_time_based=False,
                            # for fixed sampling
                            noise_vector=noise_vector,
                            epochs=epochs,
                            evaluate=evaluate,
                        ).run_loop()