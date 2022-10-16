#!/bin/bash
#SBATCH --account=debtanu.gupta
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH -w gnode03
#SBATCH --mem-per-cpu=3000

source ~/.bashrc
python3 pose.py
