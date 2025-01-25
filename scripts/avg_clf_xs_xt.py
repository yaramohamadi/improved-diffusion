# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import copy
import numpy as np

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop

# Training  
epochs = 151 
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
how_many_samples= 2500
image_size=image_size
evaluate = False # If you want to perform evaluation during training (Currently every 25 steps)

# PATHS   
# Load pretrained model from here 
load_model_path="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet64_uncond_100M_1500K.pt"
# If you are resuming a previously aborted training, include the path to the checkpoint here
resume_checkpoint= ""
# Only need this if we are evaluating FID and stuff while training
ref_dataset_npz = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'
# Fixed noise vector
noise_vector = '/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/pokemon_fixed_noise.npy'

# Load the noise vector from the .npy file
noise_vector = th.tensor(np.load(noise_vector)).to('cuda')


# ____________________ Model ____________________

for repetition in range(3):

    for p2_gamma in [0]: # TODO Quoi??

            for dataset_size in [10]:

                # The dataset you want to finetune on
                data_dir = f'/export/livia/home/vision/Ymohammadi/datasets/pokemon/pokemon{dataset_size}/' 
                source_data_dir = f'/export/livia/home/vision/Ymohammadi/datasets/imagenet_samples5000/' 

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

                for g in [0, 0.1, 0.3, 0.7, 0.9, 1]: 

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
                        strict = False if time_aware else True # TIMEAWARE
                        model.load_state_dict(checkpoint, strict = strict) # TIMEAWARE: Because we are adding some new modules  

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

                        # TIME-AWARE, in case of normal finetune, we dont freeze anything
                        if mode != 'finetune':
                            print("Doing selective unfreezing....")
                            selective_freeze_unfreeze(model, time_aware)

                        for param in model.parameters():
                            param.requires_grad = True

                        # ________________classifier-free guidance (DONT NEED TODO removes)_______________

                        # Fixed
                        guidance_scale = np.array([g for _ in range(epochs)]) # Fixed Line

                        # Where to log the training loss (File does not have to exist)
                        loss_logger=f"/export/livia/home/vision/Ymohammadi/clf_results/clf_xs_xt/data{dataset_size}/g{g}/trainlog.csv"
                        # If evaluation is true during training, where to save the FID stuff
                        eval_logger=f"/export/livia/home/vision/Ymohammadi/clf_results/clf_xs_xt/data{dataset_size}/g{g}/evallog.csv"
                        # Directory to save checkpoints in
                        checkpoint_dir = f"/export/livia/home/vision/Ymohammadi/clf_results/clf_xs_xt/data{dataset_size}/g{g}/checkpoints/"
                        # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
                        save_samples_dir= f"/export/livia/home/vision/Ymohammadi/clf_results/clf_xs_xt/data{dataset_size}/g{g}/samples/"

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
                        ).run_loop()