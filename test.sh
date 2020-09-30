#!/bin/bash

#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.error

python test.py