
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import os 
import copy
import matplotlib.pyplot as plt 

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
how_many_samples=50
image_size=image_size
evaluate = False # If you want to perform evaluation during training (Currently every 25 steps)

# PATHS   /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/
# Load pretrained model from here /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/
load_model_path="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/imagenet64_uncond_100M_1500K.pt"
# The dataset you want to finetune on
data_dir = '/home/ymbahram/scratch/pokemon/pokemon10/' 
# If you are resuming a previously aborted training, include the path to the checkpoint here
resume_checkpoint= ""
# Where to log the training loss (File does not have to exist)
loss_logger="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results_s09/trainlog.csv"
# If evaluation is true during training, where to save the FID stuff
eval_logger="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results_s09/evallog.csv"
# Directory to save checkpoints in
checkpoint_dir = "/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results_s09/checkpoints/"
# Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
save_samples_dir= "/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results_s09/samples/"
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

# ________________ Load Pretrained ____________

model_path=load_model_path
checkpoint = th.load(model_path)
model.load_state_dict(checkpoint, strict = True) 

model.to('cuda')

# ________________classifier-free guidance_______________
pretrained_model = copy.deepcopy(model)
pretrained_model.to('cuda')
guidance_scale = 0.9
pretrained_samples = "/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/results/pretrained_samples/"
classifier_free = True


pretrained_data = load_data(
    data_dir=pretrained_samples,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
)

# ________________ Train _________________ 

data = load_data(
    data_dir=data_dir,
    batch_size=batch_size,
    image_size=image_size,
    class_cond=False,
)

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
    use_ddim=False,
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
    pretrained_data=pretrained_data,
).run_loop()       

