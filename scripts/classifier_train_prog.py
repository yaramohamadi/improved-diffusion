import torch as th
from torch.optim import AdamW
import torch.nn.functional as F
from torch._utils import _unflatten_dense_tensors

import os
import blobfile as bf

from improved_diffusion.script_util import create_classifier, create_gaussian_diffusion
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import log_loss_dict

import gc
gc.collect()
th.cuda.empty_cache()


# Classifier model
image_size = 64
model_channels = 128
num_res_blocks = 3
classifier_attention_resolutions = "16,8"
classifier_use_scale_shift_norm = True
classifier_pool ="attention"

# Training
schedule_sampler="uniform" # For time-step, should it be uniform or changing based on loss function
lr=1e-4
weight_decay=0.0
iterations=700
resume_step=0 # When you do checkpoint also add this
anneal_lr=0
noised=True # Whether the images are noisy
microbatch=-1
eval_interval=10
log_interval=10
save_interval=10

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

# The dataset you want to finetune on
dataset_size = 10
batch_size=10
eval_batch_size=100


file_modes = ['spatial_v2']    # 'initial' # 'selective_freezing' 'adaptive', #'spatial', 

for file_mode in file_modes:

    classifier_pool =file_mode
    selective_freezing = False

    # Paths
    load_model_path="/home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/util_files/imagenet64_uncond_100M_1500K.pt"
    loss_logger=f"/home/ymbahram/scratch/baselines/classifier-guidance/{file_mode}/trainlog.csv"
    # If evaluation is true during training, where to save the FID stuff
    eval_logger=f"/home/ymbahram/scratch/baselines/classifier-guidance/{file_mode}/evallog.csv"
    plot_accuracy = f'/home/ymbahram/scratch/baselines/classifier-guidance/{file_mode}/accuracy_plot.png' 
    # Directory to save checkpoints in
    checkpoint_dir = f"/home/ymbahram/scratch/baselines/classifier-guidance/{file_mode}/checkpoints/"
    data_dir = f'/home/ymbahram/scratch/pokemon/pokemon10classifier/train/' 
    val_data_dir = f'/home/ymbahram/scratch/pokemon/pokemon10classifier/val/' 


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


    schedule_sampler = create_named_schedule_sampler(
                schedule_sampler, diffusion
            )

    # load model
    checkpoint = th.load(load_model_path)

    # Remove specific keys from the state_dict
    keys_to_ignore = ['out.0.weight', 'out.0.bias', 'out.2.weight', 'out.2.bias', 'out.4.bias', 'out.4.bias'] # The same name in the checkpoint is used for decoder output, but in encoderunet its used for the encoder output
    for key in keys_to_ignore:
        if key in checkpoint:
            del checkpoint[key]



        # Extract the state dictionary from the checkpoint
    checkpoint_params = checkpoint.keys()
    
    # Check for missing parameters in the model
    missing_params = [name for name, _ in classifier.named_parameters() if name not in checkpoint_params]

    # Print missing parameters
    if missing_params:
        print("Parameters not found in checkpoint:")
        for param in missing_params:
            print(param)
    else:
        print("All model parameters are found in the checkpoint.")


    classifier.load_state_dict(checkpoint, strict = False) 

    classifier.to('cuda')


    # Selective freezing
    if selective_freezing:
        for name, param in classifier.named_parameters():
            if name in checkpoint:
                param.requires_grad = False
            else:
                print(f"UnFreezing parameter: {name}")

    # Data loader

    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=True, 
        random_crop=False, # Classifier-guidance # TODO if we change this we probably should change it for the generation too so keep it untouched for now
        random_flip=False, # Classifier-guidance
        weighted_sampling=True,  # Classifier-guidance
    )




    if val_data_dir:
        val_data = load_data(
            data_dir=val_data_dir,
            batch_size=eval_batch_size,
            image_size=image_size,
            class_cond=True,
            random_crop=False, # Classifier-guidance
            random_flip=False, # Classifier-guidance
            deterministic=True, # Classifier-guidance for evaluation
            infinite=False, # Classifier-guidance for evaluation
            weighted_sampling=True,  # Classifier-guidance
        )
    else:
        val_data = None

    opt = AdamW(list(classifier.parameters()), lr=lr, weight_decay=weight_decay)



    # Train Loop
    def forward_backward_log(data_loader, prefix="train", log_path='', epoch=None):

        batch, extra = next(data_loader)
        labels = extra["y"].to('cuda')
        batch = batch.to('cuda')
        
        # Preprocess batches they are float unfort. # TODO make it cleaner do it in dataloader
        batch = batch.to(th.float32) / 255.0

        # Noisy images
        if noised:
            t, _ = schedule_sampler.sample(batch.shape[0], 'cuda')
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device='cuda')

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(microbatch, batch, labels, t)
        ):
            logits = classifier(sub_batch, timesteps=sub_t)

            loss = F.cross_entropy(logits, sub_labels, reduction="none")
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses["epoch"] = th.tensor([epoch], dtype=th.float32)

            # losses[f"{prefix}_acc@5"] = compute_top_k(   # We only have 2 classes so we don't need this bro
            #    logits, sub_labels, k=2, reduction="none"
            # )
            log_loss_dict(losses, log_path)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    classifier.zero_grad()
                loss.backward(loss * len(sub_batch) / len(batch))




    def eval_log(data_loader, prefix="val", log_path='', epoch=None):


        losses = {f"{prefix}_loss": [], f"{prefix}_acc@1": [], "epoch": th.tensor([epoch], dtype=th.float32)}
        ib = 0
        for batch, extra in data_loader:
            if ib * eval_batch_size >= 2000: # Size of dataset TODO Change and clean up
                break
            ib = ib+1
            labels = extra["y"].to('cuda')

            batch = batch.to('cuda')

            # Preprocess batches they are float unfort. # TODO make it cleaner do it in dataloader
            batch = batch.to(th.float32) / 255.0

            # Noisy images
            if noised:
                t, _ = schedule_sampler.sample(batch.shape[0], 'cuda')
                batch = diffusion.q_sample(batch, t)
            else:
                t = th.zeros(batch.shape[0], dtype=th.long, device='cuda')

            for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(microbatch, batch, labels, t)
            ):
                logits = classifier(sub_batch, timesteps=sub_t)
                loss = F.cross_entropy(logits, sub_labels, reduction="none")
                losses[f"{prefix}_loss"].append(loss.detach())
                losses[f"{prefix}_acc@1"].append(compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                ))
        
        print('eval losses')
        losses[f"{prefix}_loss"] =  th.cat(losses[f"{prefix}_loss"]).flatten()
        losses[f"{prefix}_acc@1"] =   th.cat(losses[f"{prefix}_acc@1"]).flatten()
        print(losses)
        log_loss_dict(losses, log_path)
        del losses



    # UTIL
    def set_annealed_lr(opt, base_lr, frac_done):
        lr = base_lr * (1 - frac_done)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def split_microbatches(microbatch, *args):
        bs = len(args[0])
        if microbatch == -1 or microbatch >= bs:
            yield tuple(args)
        else:
            for i in range(0, bs, microbatch):
                yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


    def compute_top_k(logits, labels, k, reduction="mean"):
        _, top_ks = th.topk(logits, k, dim=-1)
        if reduction == "mean":
            return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
        elif reduction == "none":
            return (top_ks == labels[:, None]).float().sum(dim=-1)
        

    def save(model, optim):

        params = list(model.parameters())
        def save_checkpoint(rate, params):
            state_dict = _master_params_to_state_dict(model)

            print(f"saving model {rate}...")
            if not rate:
                filename = f"model{(step+resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(step+resume_step):06d}.pt"
            with bf.BlobFile(bf.join(checkpoint_dir, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, params)
        # Remove
        with bf.BlobFile(
            bf.join(checkpoint_dir, f"opt{(step+resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(optim.state_dict(), f)

        
    def _master_params_to_state_dict(model):

        master_params = list(model.parameters())
        state_dict = model.state_dict()
        for i, (name, _) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict


    # Training

    for step in range(iterations - resume_step):
        print("step", step + resume_step)

        if anneal_lr:
            set_annealed_lr(opt, lr, (step + resume_step) / iterations)

        forward_backward_log(data, log_path=loss_logger, epoch=step + resume_step)
        opt.step()

        if val_data is not None and not step % eval_interval:
            with th.no_grad():
                classifier.eval()
                eval_log(val_data, prefix="val", log_path=eval_logger, epoch=step + resume_step)
                classifier.train()
        if not (step + resume_step) % save_interval:
            print("Saving model...")
            save(classifier, opt) # TODO Change when you wanna save model

    # Plot the line

    import matplotlib.pyplot as plt
    import pandas as pd

    # Data points
    val_data = pd.read_csv(eval_logger)
    train_data = pd.read_csv(loss_logger)

    # Plot the data
    print("Now plotting...")
    plt.figure(figsize=(10, 5))

    plt.plot(val_data['epoch'], val_data['val_acc@1'], marker='o', linestyle='-', color='b', label='Val')
    # plt.plot(train_data['epoch'], train_data['train_acc@1'], marker='o', linestyle='-', color='b', label='Train')
    plt.title("Accuracy Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_accuracy)


    print("Now removing junk checkpoints...")

    # Find index of the highest value
    highest_value_index = val_data.loc[val_data['val_acc@1'].idxmax(), 'epoch']

    # Format index to ensure it's two digits
    formatted_index = f"{int(highest_value_index):06d}"  # Pad with zeros if needed

    # Filenames to keep based on the formatted index
    keep_filenames = {f"opt{formatted_index}.pt", f"model{formatted_index}.pt"}

    # Loop through files in the directory and delete files not matching the specified format
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)
        # Delete files not in the keep_filenames set
        if filename not in keep_filenames:
            os.remove(file_path)
        else:
            print(f"Kept: {file_path}")