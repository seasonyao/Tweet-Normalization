#!/bin/bash

#SBATCH --job-name=ios0_naemb_1e4_preTweetcopy_tweetnorm
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=log/tweetnorm/%j_ios0_naemb_1e4_preTweetcopy_tweetnorm.out
#SBATCH --error=log/tweetnorm/%j_ios0_naemb_1e4_preTweetcopy_tweetnorm.error

python train_tweetNorm.py --model_name=ios0_naemb_1e4_preTweetcopy_tweetnorm