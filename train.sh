#!/bin/bash

#SBATCH --job-name=bertweet2bertweet_share
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=log/2/%j_bertweet2bertweet_share.out
#SBATCH --error=log/2/%j_bertweet2bertweet_share.error

python train_tweetNorm.py --model_name=2