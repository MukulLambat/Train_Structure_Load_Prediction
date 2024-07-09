#!/bin/bash

# Specify the SCRUM job parameters
#SBATCH --partition=gpu
#SBATCH --job-name=Mukul_Train
#SBATCH --output=Stochastic_Gradient_Descent_Model.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mukul.lambat@student.uni-siegen.de

module purge # module purge is used to unload all currently loaded modules, effectively resetting the environment to a clean state.
conda activate Train_Regression

# Run Python Script
python -nodisplay -nosplash -nodesktop -r "run('/Users/g059548/Dataset/Stochastic_Gradient_Descent_Model.py'); exit;"
