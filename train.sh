#!/bin/bash

#SBATCH --job-name=ios0_aemb_lr1e4_bert2bert_share
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=log/2/%j_ios0_aemb_lr1e4_bert2bert_share.out
#SBATCH --error=log/2/%j_ios0_aemb_lr1e4_bert2bert_share.error

python train_tweetNorm.py --model_name=ios0_aemb_lr1e4_bert2bert_share