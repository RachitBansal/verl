#!/bin/bash
#SBATCH --job-name=cbs-p2b
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbs-p2b-%A_%a.out
#SBATCH --error=logs/cbs-p2b-%A_%a.out
#SBATCH --array=0-6

# Phase 2B: Fix n_prompts=128, vary n_rollouts in {1, 2, 4, 8, 16, 32, 64}

set -euo pipefail

N_PROMPTS=128
N_ROLLOUTS_ARRAY=(1 2 4 8 16 32 64)
N_ROLLOUTS=${N_ROLLOUTS_ARRAY[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "CBS Phase 2B: n_prompts=${N_PROMPTS}, n_rollouts=${N_ROLLOUTS}"

bash experiments/critical_batch_size/scripts/cbs_sweep.sh \
    --n_prompts "${N_PROMPTS}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --phase p2b \
    --total_epochs 5 \
    --test_freq 25 \
    --save_freq 100
