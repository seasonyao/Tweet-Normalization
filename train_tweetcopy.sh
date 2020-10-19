#!/bin/bash

#SBATCH --job-name=lenall_tweetcopy
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=log/tweetcopy/%j_tweetcopy_lenall.out
#SBATCH --error=log/tweetcopy/%j_tweetcopy_lenall.error

python train_tweetCopy.py --model_name=tweetcopy_lenall
# python train_tweetCopy.py --model_name=tweetcopy_lenall_preCopy12