# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import copy
import numpy as np

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop


# ________________________ TIME-AWARE __________________________
from improved_diffusion.unet import AttentionBlock, Time_AttentionBlock

time_aware=True # TIME-AWARE


def selective_freeze_unfreeze(model, time_aware=False, target_channels=(384, 512)):
    """
    Dynamically freeze and unfreeze layers based on the time_aware setting.
    
    :param model: The UNet model with attention and time-aware modifications.
    :param time_aware: Boolean flag indicating whether time-aware attention is used.
    :param target_channels: Tuple of channels identifying time-aware blocks to remain trainable.
    """
    
    # Step 1: Freeze all parameters in the model

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Helper function to unfreeze a specific block
    def unfreeze_block(block):
        for param in block.parameters():
            param.requires_grad = True

    # Step 2: Iterate through all blocks to find target blocks
    for block in model.input_blocks:
        for sub_layer in block:
            if hasattr(sub_layer, 'channels'):
                if time_aware:
                    # Unfreeze time-aware blocks with target channels
                    if sub_layer.channels in target_channels and isinstance(sub_layer, Time_AttentionBlock):
                        unfreeze_block(sub_layer)
                else:
                    # Unfreeze regular attention blocks when time-aware is False
                    if isinstance(sub_layer, AttentionBlock):
                        unfreeze_block(sub_layer)
    
    # Check the middle block
    for sub_layer in model.middle_block:
        if hasattr(sub_layer, 'channels'):
            if time_aware:
                if sub_layer.channels in target_channels and isinstance(sub_layer, Time_AttentionBlock):
                    unfreeze_block(sub_layer)
            else:
                if isinstance(sub_layer, AttentionBlock):
                    unfreeze_block(sub_layer)

    # Iterate through output blocks similarly
    for block in model.output_blocks:
        for sub_layer in block:
            if hasattr(sub_layer, 'channels'):
                if time_aware:
                    if sub_layer.channels in target_channels and isinstance(sub_layer, Time_AttentionBlock):
                        unfreeze_block(sub_layer)
                else:
                    if isinstance(sub_layer, AttentionBlock):
                        unfreeze_block(sub_layer)

    print(f"{'Time-aware' if time_aware else 'Regular'} attention blocks are now unfrozen for training.")# ______________________________________________________________
# ____________________________________________________________



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

        modes = ['finetune'] # ''] # a3ft,  'attention_finetune', 

        for mode in modes: 

            print("*"*20)
            print('Mode : ', mode)
            print("*"*20)
            
            # TIMEAWARE
            time_aware = True if mode =='a3ft' else False

            for dataset_size in [10]:

                # The dataset you want to finetune on
                data_dir = f'/home/ymbahram/scratch/pokemon/pokemon{dataset_size}/' 

                data = load_data(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    image_size=image_size,
                    class_cond=False,
                )

                for g in [0.1]: # 0, 0.01, 0.05, 

                    for gamma in [10]:

                        print("*"*20)
                        print(f"guidance is {g} gamma is {gamma} mode is {mode}")
                        print("*"*20)

                        if g == 0:
                            if gamma in [0.1, 1, 10]:
                                continue

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
                            time_aware=time_aware, # TIME-AWARE
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
                            time_aware=False, # TIME-AWARE
                        )

                        # ________________ Load Pretrained ____________

                        checkpoint = th.load(load_model_path)
                        strict = False if time_aware else True # TIMEAWARE
                        model.load_state_dict(checkpoint, strict = strict) # TIMEAWARE: Because we are adding some new modules  
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
                            p2_gamma=gamma, 
                        )

                        # TIME-AWARE, in case of normal finetune, we dont freeze anything
                        if mode != 'finetune':
                            print("Doing selective unfreezing....")
                            selective_freeze_unfreeze(model, time_aware)

                        for param in model.parameters():
                            param.requires_grad = True

                        for param in pretrained_model.parameters():
                            param.requires_grad = False

                        # ________________classifier-free guidance (DONT NEED TODO removes)_______________

                        # Fixed
                        guidance_scale = np.array([g for _ in range(epochs)]) # Fixed Line

                        # Where to log the training loss (File does not have to exist)
                        loss_logger=f"/home/ymbahram/scratch/baselines_avg/classifier-free/data{dataset_size}/{mode}/g{g}_gamma{p2_gamma}_repetition{repetition}/trainlog.csv"
                        # If evaluation is true during training, where to save the FID stuff
                        eval_logger=f"/home/ymbahram/scratch/baselines_avg/classifier-free/data{dataset_size}/{mode}/g{g}_gamma{p2_gamma}_repetition{repetition}/evallog.csv"
                        # Directory to save checkpoints in
                        checkpoint_dir = f"/home/ymbahram/scratch/baselines_avg/classifier-free/data{dataset_size}/{mode}/g{g}_gamma{p2_gamma}_repetition{repetition}/checkpoints/"
                        # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
                        save_samples_dir= f"/home/ymbahram/scratch/baselines_avg/classifier-free/data{dataset_size}/{mode}/g{g}_gamma{p2_gamma}_repetition{repetition}/samples/"

                        # ________________ Train _________________ 

                        schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

                        TrainLoop(
                            model=model,
                            diffusion=diffusion,
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
                            pretrained_model=pretrained_model,
                            guidance_scale=guidance_scale,
                            clf_time_based=False,
                            # for fixed sampling
                            noise_vector=noise_vector,
                            epochs=epochs,
                        ).run_loop()