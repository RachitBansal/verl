#!/bin/bash
#SBATCH --job-name=cbs-p2c
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbs-p2c-%A_%a.out
#SBATCH --error=logs/cbs-p2c-%A_%a.out
#SBATCH --array=0-5

# Phase 2C: Iso-batch comparison. Total batch = 2048 = n_prompts * n_rollouts
# Decompositions: (2048,1) (1024,2) (512,4) (256,8) (128,16) (64,32)

set -euo pipefail

N_PROMPTS_ARRAY=(2048 1024 512 256 128 64)
N_ROLLOUTS_ARRAY=(1    2    4   8   16  32)

N_PROMPTS=${N_PROMPTS_ARRAY[$SLURM_ARRAY_TASK_ID]}
N_ROLLOUTS=${N_ROLLOUTS_ARRAY[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "CBS Phase 2C (iso-batch=2048): n_prompts=${N_PROMPTS}, n_rollouts=${N_ROLLOUTS}"

bash experiments/critical_batch_size/scripts/cbs_sweep.sh \
    --n_prompts "${N_PROMPTS}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --phase p2c \
    --total_epochs 5 \
    --test_freq 25 \
    --save_freq 100
