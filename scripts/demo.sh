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
#SBATCH --job-name=FTR1
#SBATCH -o slurm_outputs/batch_inf_%j.out

# srun -n 1 -c 4 --mem=32G --gres=gpu:1 -C '(turing|volta)' --pty bash
# source env/bin/activate
# conda activate PIDM

python demo_tempo2.py


# FT020 - finetuned model start 0, step 20
# FT01 - finetuned model start 0, step 1
# FTR1 - finetuned model start R, step 1