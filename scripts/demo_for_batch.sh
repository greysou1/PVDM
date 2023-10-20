#!/bin/bash

# ======== CPU Config ======== 
#SBATCH -n 1
#SBATCH -c 4 
#SBATCH --mem=64G 

# ======== GPU Config ======== 
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem24

# ======== Slurm config ======== 
#SBATCH --job-name=forbatch
#SBATCH -o slurm_outputs/batch_inf_%j.out

# srun -n 1 -c 4 --mem=32G --gres=gpu:1 -C '(turing|volta)' --pty bash
# source env/bin/activate
# conda activate PIDM

python demo_for_batch.py
