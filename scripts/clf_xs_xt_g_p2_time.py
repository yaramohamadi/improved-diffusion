import os
import yaml
import socket

import copy
import numpy as np

import torch as th
from improved_diffusion.script_util import create_model, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop


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
evaluate = True # If you want to save npz to perform evaluation later (FID and stuff)

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





# ____________________ TIME SCHEDULE __________________


import numpy as np
from improved_diffusion import gaussian_diffusion as gd

def compute_snr_curve(noise_schedule, steps, timestep_respacing, p2_gamma):
    def space_timesteps(num_timesteps, section_counts):
        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim"):])
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(
                    f"cannot create exactly {num_timesteps} steps with an integer stride"
                )
            section_counts = [int(x) for x in section_counts.split(",")]
        size_per = num_timesteps // len(section_counts)
        extra = num_timesteps % len(section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        return set(all_steps)

    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    use_timesteps = set(space_timesteps(steps, timestep_respacing))
    timestep_map = []
    original_num_steps = len(betas)

    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    betas = np.array(new_betas)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    snr = 1.0 / (1 - alphas_cumprod) - 1
     
    curve = 1 / (1 + snr)**p2_gamma

    # This is the reverse of P-2 weighting! because we are guiding the source model (g=0) to the target (g=1), 
    return 1 - curve

# Example usage
# noise_schedule = "cosine"
# steps = 4000
# timestep_respacing = "ddim50"
p2_gamma_values = [0.1, 0.5, 2.0, 10, 50]

time_curves = {
    "50": compute_snr_curve(noise_schedule, diffusion_steps, timestep_respacing, 50),
    "10":compute_snr_curve(noise_schedule, diffusion_steps, timestep_respacing, 10),
    "2":compute_snr_curve(noise_schedule, diffusion_steps, timestep_respacing, 2),
    "0.5": compute_snr_curve(noise_schedule, diffusion_steps, timestep_respacing, 0.5),
    "0.1": compute_snr_curve(noise_schedule, diffusion_steps, timestep_respacing, 0.1),
    "0.01": compute_snr_curve(noise_schedule, diffusion_steps, timestep_respacing, 0.01),
}

# ____________________ Model ____________________


for repetition in range(1):

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

                for t_gamma, t_curve in time_curves.items():  # 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1 0.1  

                    print(t_curve)


                    for gamma in [0]:

                        print("*"*20)
                        print(f"time guidance gamma is {t_gamma} gamma is {gamma}")
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

                        # Time based guidance
                        guidance_scale = t_curve
                        clf_time_based = True

                        # Where to log the training loss (File does not have to exist)
                        loss_logger = os.path.join(base_path, f"clf_results/clf_xs_xt/time_p2/t_gamma_reverse{t_gamma}/trainlog.csv")
                        # If evaluation is true during training, where to save the FID stuff
                        eval_logger = os.path.join(base_path, f"clf_results/clf_xs_xt/time_p2/t_gamma_reverse{t_gamma}/evallog.csv")
                        # Directory to save checkpoints in 
                        checkpoint_dir = os.path.join(base_path, f"clf_results/clf_xs_xt/time_p2/tmpcheckpoints/")
                        # Whenever you are saving checkpoints, a batch of images are also sampled, where to produce these images
                        save_samples_dir= os.path.join(base_path, f"clf_results/clf_xs_xt/time_p2/t_gamma_reverse{t_gamma}/samples/")

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
                            clf_time_based=True,
                            # for fixed sampling
                            noise_vector=noise_vector,
                            epochs=epochs,
                            evaluate=evaluate,
                        ).run_loop()