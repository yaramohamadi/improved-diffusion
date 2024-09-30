
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import os 
import copy
import matplotlib.pyplot as plt 
import numpy as np

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop

# Training  
batch_size=10
schedule_sampler="uniform" # For time-step, should it be uniform or changing based on loss function
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
how_many_samples= 2503
image_size=image_size
evaluate = False # If you want to perform evaluation during training (Currently every 25 steps)

# PATHS   /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/
# Load pretrained model from here /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/
load_model_path="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/imagenet64_uncond_100M_1500K.pt"
# If you are resuming a previously aborted training, include the path to the checkpoint here
resume_checkpoint= ""
# Only need this if we are evaluating FID and stuff while training
ref_dataset_npz = '/home/ymbahram/scratch/pokemon/pokemon_64x64.npz'



# ____________________ Model ____________________


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
)

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
)

# For guidance scheduler
def create_scaled_line(start, end, a=0):
    x = np.linspace(0, 1, 50)
    if a == 0:
        y = x
    else:
        y = (np.exp(a * x) - 1) / (np.exp(a) - 1)    
    scaled_y = start + (end - start) * y
    return scaled_y


for dataset_size in [10, 100, 700, 2503]:

    # The dataset you want to finetune on
    data_dir = f'/home/ymbahram/scratch/pokemon/pokemon{dataset_size}/' 

    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
    )

    for g, g_name in {
        # Fixed
        0:'0', 0.05:'0_05', 0.1:'0_1', 0.2: '0_2'
        }.items():

        # ________________ Load Pretrained ____________

        model_path=load_model_path
        checkpoint = th.load(model_path)
        model.load_state_dict(checkpoint, strict = True) 

        model.to('cuda')

        # ________________classifier-free guidance_______________
        pretrained_model = copy.deepcopy(model)
        pretrained_model.to('cuda')
        classifier_free = True
        clf_time_based = False 

        # Imagine we are training for 200 epochs max 
        # Fixed
        guidance_scale = np.array([g for _ in range(201)]) # Fixed Line

        # Where to log the training loss (File does not have to exist)
        loss_logger=f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data{dataset_size}/{g_name}/trainlog.csv"
        # If evaluation is true during training, where to save the FID stuff
        eval_logger=f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data{dataset_size}/{g_name}/evallog.csv"
        # Directory to save checkpoints in
        checkpoint_dir = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data{dataset_size}/{g_name}/checkpoints/"
        # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
        save_samples_dir= f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_trg_results/fixed_guidance_dataset/data{dataset_size}/{g_name}/samples/"

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
            # For evaluating
            evaluate = evaluate,
            eval_logger = eval_logger,
            reference_dataset_dir=ref_dataset_npz, # If sampling is true, then Evaluation will be done here,
            eval_func=None,
            # For classifier-free guidanace
            pretrained_model=pretrained_model,
            guidance_scale=guidance_scale,
            clf_time_based=clf_time_based
        ).run_loop()