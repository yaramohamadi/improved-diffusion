from improved_diffusion.script_util import create_classifier, create_gaussian_diffusion
import torch as th

# Classifier train 
image_size = 64
model_channels = 128
num_res_blocks = 3
classifier_attention_resolutions = "16,8"
classifier_use_scale_shift_norm = True
classifier_pool ="attention"

# Diffusion
diffusion_steps=4000
learn_sigma=True
sigma_small=False
noise_schedule="cosine"
use_kl=False
predict_xstart=False
rescale_timesteps=True
rescale_learned_sigmas=True
timestep_respacing="ddim50"

# Paths
load_model_path="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet64_uncond_100M_1500K.pt"

# Classifier-guidance
classifier = create_classifier(
    image_size = image_size,
    model_channels = model_channels,
    num_res_blocks = num_res_blocks,
    classifier_attention_resolutions = classifier_attention_resolutions,
    classifier_use_scale_shift_norm = classifier_use_scale_shift_norm,
    classifier_pool =classifier_pool,
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

# load model
checkpoint = th.load(load_model_path)

# Remove specific keys from the state_dict
keys_to_ignore = ['out.0.weight', 'out.0.bias'] # The same name in the checkpoint is used for decoder output, but in encoderunet its used for the encoder output
for key in keys_to_ignore:
    if key in checkpoint:
        del checkpoint[key]

classifier.load_state_dict(checkpoint, strict = False) 

classifier.to('cuda')