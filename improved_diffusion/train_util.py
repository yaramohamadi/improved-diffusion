# I have changed the code to exclude everything that is for distributed learning

import copy
import functools
import os
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt 

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        loss_logger="trainlog.csv",
        checkpoint_dir = "./checkpoints/",
        sample = False,
        use_ddim=False,
        save_samples_dir="./samples/",
        how_many_samples=50, # For sampling mid training
        image_size=64
    ):
        self.image_size=image_size
        self.save_samples_dir = save_samples_dir
        self.sample = sample
        self.how_many_samples=how_many_samples
        self.use_ddim = use_ddim
        self.checkpoint_dir = checkpoint_dir
        self.loss_logger = loss_logger
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # REMOVED * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        
        if self.resume_step: # This is set automatically if loading the checkpoint
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
            

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # REMOVED 
            print(f"loading model from checkpoint: {resume_checkpoint}...")

            with bf.BlobFile(resume_checkpoint, "rb") as f:
                data = f.read()
            self.model.load_state_dict(
                # REMOVED                             
                th.load(io.BytesIO(data), **kwargs)
            )
        # REMOVED

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # REMOVED
            print(f"loading EMA from checkpoint: {ema_checkpoint}...")
    
            with bf.BlobFile(ema_checkpoint, "rb") as f: 
                data = f.read()
            state_dict = th.load(io.BytesIO(data))

            ema_params = self._state_dict_to_master_params(state_dict)
        # Remove
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")

            with bf.BlobFile(opt_checkpoint, "rb") as f: 
                data = f.read()
            state_dict = th.load(io.BytesIO(data))
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        def loop():
            while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                yield # This terminology is for working with tqdm and infinite while loops 

        for _ in tqdm(loop()):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.save_interval == 0:
                self.save()
                if self.sample: # Added this for sampling
                    self.model.eval()
                    self.samplefunc()
                    self.model.train()
            self.step += 1
            
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        # self.log_step()

    def forward_backward(self, batch, cond):
        """
        Compute loss by calling diffusion.training_losses 
        In order to get output also we should output something from training_losses
        """
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to('cuda') # REMOVED
            micro_cond = {
                k: v[i : i + self.microbatch].to('cuda') # REMOVED
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], 'cuda') # REMOVED

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}, self.loss_logger
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        # self._log_grad_norm() # Had error, there is no grad
        self._anneal_lr() 
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        # self._log_grad_norm()  
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        print(self.master_params)
        for p in self.master_params:

            sqsum += (p.grad ** 2).sum().item()
        print("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        print("step", self.step + self.resume_step)
        if self.use_fp16:
            print("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)

            print(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        # Remove
        with bf.BlobFile(
            bf.join(self.checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    
    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

    # I CREATED THIS FUNCTION, sample a batch everytime you save checkpoints (Changed)
    def samplefunc(self):

        print(f"sampling {self.how_many_samples} images")
        sample_fn = (self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop)
        sample = sample_fn(
            self.model,
            (self.how_many_samples, 3, self.image_size , self.image_size),
            clip_denoised=True,
            model_kwargs={}, # This is not needed, just class conditional stuff
            progress=True
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous().cpu().numpy()

        # Save images
        path = os.path.join(self.save_samples_dir, str(self.step+self.resume_step))
        os.makedirs(path, exist_ok=True)
        for sidx, s in enumerate(sample):
            plt.imsave(os.path.join(path, f'{sidx}.jpg'), s)
    

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
          return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


# YO YO YO Im changin here
def log_loss_dict(diffusion, ts, losses, loss_logger):

    # Check if the file exists
    file_exists = os.path.isfile(loss_logger)
    with open(loss_logger, 'a', newline='') as csvfile:
        fieldnames = losses.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writeheader()
        losses_int = {key: tensor.mean().item() for key, tensor in losses.items()}
        # Write the data
        writer.writerow(losses_int)
        
    #for key, values in losses.items():
    #    print(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(),  values.detach().cpu().numpy()):         # CHANGED THIS, Dont need it
        #    quartile = int(4 * sub_t / diffusion.num_timesteps)
        #    print(f"{key}_q{quartile}", sub_loss)
