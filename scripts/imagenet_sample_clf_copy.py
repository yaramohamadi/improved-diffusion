
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

from improved_diffusion.script_util import create_model, create_gaussian_diffusion
import os 
import matplotlib.pyplot as plt 
import torch as th
import numpy as np
import copy
from tqdm import tqdm

# Sampling 
batch_size=10
timestep_respacing="ddim50"
use_ddim=True
num_samples=50
clip_denoised=True
source_model_path = "/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/pretrained_models/imagenet64_uncond_100M_1500K.pt"

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
# ________________classifier-free guidance_______________

source_model = copy.deepcopy(model)

checkpoint = th.load(source_model_path)
source_model.load_state_dict(checkpoint)

model.to('cuda')
source_model.to('cuda')
model.eval()
source_model.eval()

classifier_free = True

# ________________ Sample _________________ 

for g in [1.2]: # Fixed guidances I want to try

    save_samples_dir = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/fixed_guidance/1/clf_sampling/diff_guidance/{g}"
    os.makedirs(save_samples_dir, exist_ok=True)

    print(f'===============================Now onto {g}+++++++++++++++++++++++++++')

    for epoch in tqdm(['075']): # Sample models from each time-step

        output_dir = os.path.join(save_samples_dir, epoch)
        os.makedirs(output_dir, exist_ok=True)
        # Load model
        model_path = f"/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/clf_results/fixed_guidance/1/checkpoints/model000{epoch}.pt"
        checkpoint = th.load(model_path)
        model.load_state_dict(checkpoint)
        
        all_images = []
        i = 0

        while len(all_images) * batch_size < num_samples:

            # print(f"sampling {batch_size} images")
            sample_fn = (diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop)
            sample = sample_fn(
                model,
                (batch_size, 3, image_size , image_size),
                source_model=source_model, # classifier-free guidance
                guidance=g,
                clip_denoised=True,
                model_kwargs={}, # This is not needed, just class conditional stuff
                progress=False
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous().cpu().numpy()

            # Save images
            if i <5:
                for sidx, s in enumerate(sample):
                    #print(output_dir)
                    #print(os.path.join(output_dir, f'{sidx + i*batch_size}.jpg'))
                    plt.imsave(os.path.join(output_dir, f'{sidx + i*batch_size}.jpg'), s)
            all_images.extend(sample)
            i = i+1

        #all_images = all_images[: num_samples]
        #sample_path = os.path.join(save_samples_dir, f"samples.npz")
        #np.savez(sample_path, all_images)
        #print("sampling complete")
