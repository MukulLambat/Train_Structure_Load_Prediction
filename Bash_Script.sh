#!/bin/bash

# Specify the SCRUM job parameters
#SBATCH --partition=gpu
#SBATCH --job-name=CNN
#SBATCH --output=CNN.txt
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mukul.lambat@student.uni-siegen.de
module load GpuModules
eval "$(conda shell.bash hook)"
conda deactivate
conda activate Train_Regression

# Run Python Script
python /work/ws-tmp/g059548-ML_Project/CNN.py
