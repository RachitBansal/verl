#!/bin/bash
#SBATCH --job-name=create-math-subset
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/create-math-subset-%j.out
#SBATCH --error=logs/create-math-subset-%j.out

mkdir -p logs
module purge
module load Mambaforge

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

python /n/netscratch/dam_lab/Lab/brachit/rollouts/verl/scripts/create_math_5k_subset.py

