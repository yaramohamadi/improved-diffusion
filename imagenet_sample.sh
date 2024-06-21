#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8    # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --mem=10000M
#SBATCH --time=00:10:00
#SBATCH --account=def-hadi87

source /home/ymbahram/.bash_profile
module load python/3.11.5
source /home/ymbahram/projects/def-hadi87/ymbahram/gputorch/bin/activate

python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/finetuning10_cfguidance.py

