#!/bin/bash
#SBATCH --job-name=cbs-p4
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cbs-p4-%A_%a.out
#SBATCH --error=logs/cbs-p4-%A_%a.out
#SBATCH --array=0-17

# Phase 4: Scale-up and ablation experiments
#
# Block 1 (0-5):   OLMo2-1B on MATH (harder task), key batch sizes
# Block 2 (6-11):  OLMo2-1B with REINFORCE++ advantage estimator
# Block 3 (12-17): OLMo2-1B with different learning rates (LR sensitivity)

set -euo pipefail

# Experiment configurations
N_PROMPTS_ALL=(64 128 256 512 1024 2048  64 128 256 512 1024 2048  128 128 128 512 512 512)
N_ROLLOUTS_ALL=(8  8   8   8   8    8    8  8   8   8   8    8     8   8   8   8   8   8)

# Phase tags
PHASE_TAGS=( \
    p4_math p4_math p4_math p4_math p4_math p4_math \
    p4_rpp p4_rpp p4_rpp p4_rpp p4_rpp p4_rpp \
    p4_lr5e7 p4_lr1e6 p4_lr5e6 p4_lr5e7 p4_lr1e6 p4_lr5e6 \
)

# Advantage estimator per experiment
ADV_EST_ALL=( \
    grpo grpo grpo grpo grpo grpo \
    reinforce_plus_plus reinforce_plus_plus reinforce_plus_plus reinforce_plus_plus reinforce_plus_plus reinforce_plus_plus \
    grpo grpo grpo grpo grpo grpo \
)

# Learning rates per experiment
LR_ALL=( \
    1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 \
    1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 \
    5e-7 1e-6 5e-6 5e-7 1e-6 5e-6 \
)

IDX=$SLURM_ARRAY_TASK_ID
N_PROMPTS=${N_PROMPTS_ALL[$IDX]}
N_ROLLOUTS=${N_ROLLOUTS_ALL[$IDX]}
PHASE_TAG=${PHASE_TAGS[$IDX]}
ADV_EST=${ADV_EST_ALL[$IDX]}
LR=${LR_ALL[$IDX]}

mkdir -p logs

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "CBS Phase 4 [$IDX]: ${PHASE_TAG}"
echo "  n_prompts=${N_PROMPTS}, n_rollouts=${N_ROLLOUTS}, adv=${ADV_EST}, lr=${LR}"

# Use MATH data for p4_math block
EXTRA_ARGS="algorithm.adv_estimator=${ADV_EST}"
if [[ "${PHASE_TAG}" == "p4_math" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS}"
    bash experiments/critical_batch_size/scripts/cbs_sweep.sh \
        --n_prompts "${N_PROMPTS}" \
        --n_rollouts "${N_ROLLOUTS}" \
        --phase "${PHASE_TAG}" \
        --lr "${LR}" \
        --data_dir "/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_math" \
        --total_epochs 10 \
        --test_freq 25 \
        --save_freq 200 \
        --extra_args "${EXTRA_ARGS}"
else
    bash experiments/critical_batch_size/scripts/cbs_sweep.sh \
        --n_prompts "${N_PROMPTS}" \
        --n_rollouts "${N_ROLLOUTS}" \
        --phase "${PHASE_TAG}" \
        --lr "${LR}" \
        --total_epochs 5 \
        --test_freq 25 \
        --save_freq 200 \
        --extra_args "${EXTRA_ARGS}"
fi
