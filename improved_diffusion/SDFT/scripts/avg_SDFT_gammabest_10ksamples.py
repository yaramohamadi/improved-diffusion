# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import copy
import numpy as np

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop

import gc
gc.collect()
th.cuda.empty_cache()


# ______________________ Sample related ________________________
how_many_samples= 10000 


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
how_many_samples= 10000 
image_size=image_size
evaluate = False # If you want to perform evaluation during training (Currently every 25 steps)

# PATHS   
load_model_path="/home/ymbahram/scratch/util_files/imagenet64_uncond_100M_1500K.pt"
noise_vector = '/home/ymbahram/scratch/util_files/pokemon_fixed_noise.npy'

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

# Load the noise vector from the .npy file
noise_vector = th.tensor(np.load(noise_vector))  # Load on CPU
noise_vector = noise_vector.to('cuda')  # Transfer to GPU if memory allows


lambda_auxs = [0.001] 
lambda_distils = [0.001]
# SDFT: Output from auxiliary input drastically collapses in smaller timesteps therefore larger gamma (Less influence in smaller timesteps)
gamma_auxs = [
    0.1]
gamma_distils = [0.1]

for repetition in range(3):

    for p2_gamma in [0]: # TODO Quoi??

        for lambda_distil, lambda_aux in zip(lambda_distils, lambda_auxs): # SDFT: We assume that these two hyperparameters should be the same, just like in the paper
            for gamma_aux, gamma_distil in zip(gamma_auxs, gamma_distils):

                    if lambda_distil == 0:
                        if gamma_distil == 10: # For lambda = 0 its gonna just be like fine-tuning so the 1gamma value does not matter here
                            gamma_distil = 9999
                        else:
                            continue
                    
                    print("*"*20)
                    print(f"lambda_distil: {lambda_distil}, lambda_aux: {lambda_aux}, gamma_aux: {gamma_aux}, gamma_distil: {gamma_distil}")
                    print("*"*20)

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
                        SDFT=True,# For SDFT
                        gamma_distil=gamma_distil,# For SDFT
                        gamma_aux=gamma_aux,# For SDFT
                        lambda_distil=lambda_distil, # for SDFT
                        lambda_aux=lambda_aux, # for SDFT
                        p2_gamma=p2_gamma,
                    )

                    for dataset_size in [10]: 
                        
                        # The dataset you want to finetune on
                        data_dir = f'/home/ymbahram/scratch/datasets/pokemon/pokemon{dataset_size}/' 

                        data = load_data(
                            data_dir=data_dir,
                            batch_size=batch_size,
                            image_size=image_size,
                            class_cond=False,
                        )

                        for g, g_name in {
                            # Fixed
                            0.0:'0' 
                            }.items():

                            print("*"*20)
                            print(f"fixed guidance is {g_name}")
                            print("*"*20)

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
                            guidance_scale = np.array([g for _ in range(epochs)]) # Fixed Line


                            checkpoint_dir = f"/home/ymbahram/scratch/baselines_avg/SDFT/data{dataset_size}/p2_gamma{p2_gamma}_repeat{repetition}/lambdas{lambda_distil}_gammas{gamma_distil}/checkpoints/"
                            # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
                            save_samples_dir= f"/home/ymbahram/scratch/baselines_avg/SDFT/data{dataset_size}/p2_gamma{p2_gamma}_repeat{repetition}/lambdas{lambda_distil}_gammas{gamma_distil}/samples/"

                            # ________________ Sample _________________ 


                                                
                            print(f"sampling {how_many_samples} images")
                            sample_fn = (diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop)

                            # Create folder
                            im_path = os.path.join(save_samples_dir)
                            os.makedirs(im_path, exist_ok=True)

                            all_images = []
                            for ind, _ in tqdm(enumerate(range(0, how_many_samples + batch_size - 1, batch_size))):

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
                            
                            sample_path = os.path.join(save_samples_dir, f"samples_10k.npz")
                            np.savez(sample_path, all_images)
                            print("sampling complete")


                            
                            

