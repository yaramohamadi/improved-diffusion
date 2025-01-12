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
epochs = 201
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

for repetition in range(1, 3):

    for p2_gamma in [0]: # TODO Quoi??

        for lambda_1 in [0.1]: # , 0.5, 1
            
            for lambda_2 in [0.5]: # 0.1, 1
                
                for lambda_3 in [0.08]:

                    for dataset_size in [10]:

                        # The dataset you want to finetune on
                        data_dir = f'/home/ymbahram/scratch/pokemon/pokemon{dataset_size}/' 

                        data = load_data(
                            data_dir=data_dir,
                            batch_size=batch_size,
                            image_size=image_size,
                            class_cond=False,
                        )

                        for g in [0]: # 0, 0.01, 0.05, 

                            for gamma in [0]:

                                print("*"*20)
                                print(f"guidance is {g} gamma is {gamma} lambda_1 is {lambda_1} lambda_2 is {lambda_2} lambda_3 is {lambda_3}")
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
                                    p2_gamma=gamma, 
                                    lambda_1=lambda_1,
                                    lambda_2=lambda_2,
                                    lambda_3=lambda_3,
                                )

                                for param in model.parameters():
                                    param.requires_grad = True

                                for param in pretrained_model.parameters():
                                    param.requires_grad = False

                                # ________________classifier-free guidance (DONT NEED TODO removes)_______________


                                # Where to log the training loss (File does not have to exist) ddpm-pa _repetition{repetition}
                                loss_logger=f"/home/ymbahram/scratch/baselines_avg/ddpm-pa/data{dataset_size}/l1_{lambda_1}_l2_{lambda_2}_l3_{lambda_3}_repeat_{repetition}/trainlog.csv"
                                # If evaluation is true during training, where to save the FID stuff
                                eval_logger=f"/home/ymbahram/scratch/baselines_avg/ddpm-pa/data{dataset_size}/l1_{lambda_1}_l2_{lambda_2}_l3_{lambda_3}_repeat_{repetition}/evallog.csv"
                                # Directory to save checkpoints in
                                checkpoint_dir = f"/home/ymbahram/scratch/baselines_avg/ddpm-pa/data{dataset_size}/l1_{lambda_1}_l2_{lambda_2}_l3_{lambda_3}_repeat_{repetition}/checkpoints/"
                                # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
                                save_samples_dir= f"/home/ymbahram/scratch/baselines_avg/ddpm-pa/data{dataset_size}/l1_{lambda_1}_l2_{lambda_2}_l3_{lambda_3}_repeat_{repetition}/samples/"

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
                                    eval_logger=eval_logger,
                                    checkpoint_dir = checkpoint_dir,
                                    # next 4 For sampling
                                    sample = True, # Doing sampling for a batch in training every time saving
                                    use_ddim=use_ddim,
                                    save_samples_dir=save_samples_dir,
                                    how_many_samples=how_many_samples,
                                    image_size=image_size,
                                    # For classifier-free guidanace (We dont need these, TODO delete later)
                                    pretrained_model=pretrained_model,
                                    clf_time_based=False,
                                    # for fixed sampling
                                    noise_vector=noise_vector,
                                    epochs=epochs,
                                ).run_loop()