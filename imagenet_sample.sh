#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8    # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --mem=30000M
#SBATCH --time=30:00:00
#SBATCH --account=def-hadi87
#SBATCH --mail-user=yara.mohammadi-bahram.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL

source /home/ymbahram/.bash_profile
module load python/3.11.5
source /home/ymbahram/projects/def-hadi87/ymbahram/gputorch/bin/activate
# pip install -e .

# python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/finetuning10_cfguidance.py
#python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/finetuning_dataset_clf.py
# python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/evaluation/evaluate.py
# python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/finetuning_dataset_clf.py
# python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/finetuning_samesample.py

# python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/imagenet_sample.py
python /home/ymbahram/projects/def-hadi87/ymbahram/improved_diffusion/classifier-free-guidance/scripts/imagenet_sample_clf_samesample.py