#!/bin/bash
#SBATCH --job-name=cbs-p1
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/cbs-p1-%A_%a.out
#SBATCH --error=logs/cbs-p1-%A_%a.out
#SBATCH --array=0-6

# Phase 1: Vary n_prompts with fixed n_rollouts=8
# Array index maps to n_prompts values: 32, 64, 128, 256, 512, 1024
# Vary clip ratio: 0.0, 1.0

set -euo pipefail

N_ROLLOUTS=8
N_PROMPTS_ARRAY=(1024 256 32)
N_PROMPTS=${N_PROMPTS_ARRAY[$SLURM_ARRAY_TASK_ID]}
CLIP_RATIO_ARRAY=(0.0 1.0)
CLIP_RATIO=${CLIP_RATIO_ARRAY[$SLURM_ARRAY_TASK_ID]}

# mini-batch size: use the total batch or 256, whichever is smaller,
# to keep gradient accumulation steps reasonable
if [ "$N_PROMPTS" -le 256 ]; then
    MINI_BATCH_SIZE=$N_PROMPTS
else
    MINI_BATCH_SIZE=256
fi

mkdir -p logs

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "============================================================"
echo "CBS Phase 1: Array task ${SLURM_ARRAY_TASK_ID}"
echo "  n_prompts=${N_PROMPTS}, n_rollouts=${N_ROLLOUTS}, clip_ratio=${CLIP_RATIO}, mini_batch_size=${MINI_BATCH_SIZE}"
echo "  Started at: $(date)"
echo "============================================================"

bash experiments/critical_batch_size/scripts/cbs_sweep.sh \
    --n_prompts "${N_PROMPTS}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --clip_ratio "${CLIP_RATIO}" \
    --mini_batch_size "${MINI_BATCH_SIZE}" \
    --phase p1 \
    --total_epochs 5 \
    --test_freq 25 \
    --save_freq 100

echo "============================================================"
echo "CBS Phase 1: Completed at $(date)"
echo "============================================================"
