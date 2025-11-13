#!/bin/bash

#SBATCH --job-name=olmo2-evaluation
#SBATCH --output=logs/olmo2-math-sweep-%A_%a.out
#SBATCH --error=logs/olmo2-math-sweep-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=500GB		
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_grads
#SBATCH --array=0,1,2,3

set -e  # Exit on error
set -u  # Exit on undefined variable

# Base model configuration
BASE_MODEL_PATH="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1-50B"
BASE_MODEL_NAME="olmo2-1b"

# Checkpoint steps to evaluate
CHECKPOINT_STEPS=(
    "step14000"
    "step5000"
    "step10000"
    "step22000"
    # "step1000"
    # "step5000"
    # "step10000"
    # "step15000"
    # "step20000"
)

# Number of samples to evaluate (for top-k sampling)
N_SAMPLES_LIST=(
    1      # Greedy-like with temperature
    # 8      # Top-8 sampling
    # 32     # Top-32 sampling
    # 128    # Top-128 sampling
)

# Select checkpoint and n_samples based on SLURM_ARRAY_TASK_ID

# Get total combinations
NUM_CHECKPOINTS=${#CHECKPOINT_STEPS[@]}
NUM_SAMPLES=${#N_SAMPLES_LIST[@]}
TOTAL_COMBINATIONS=$((NUM_CHECKPOINTS * NUM_SAMPLES))

# Get array index
ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}

if [ "${ARRAY_IDX}" -ge "${TOTAL_COMBINATIONS}" ]; then
    echo "SLURM_ARRAY_TASK_ID (${ARRAY_IDX}) exceeds total combinations (${TOTAL_COMBINATIONS})"
    exit 1
fi

# Map SLURM_ARRAY_TASK_ID to checkpoint and n_samples
STEP_IDX=$((ARRAY_IDX / NUM_SAMPLES))
N_SAMPLES_IDX=$((ARRAY_IDX % NUM_SAMPLES))

SELECTED_STEP="${CHECKPOINT_STEPS[${STEP_IDX}]}"
SELECTED_N_SAMPLES="${N_SAMPLES_LIST[${N_SAMPLES_IDX}]}"

# For downstream code to use
STEP="${SELECTED_STEP}"
N_SAMPLES="${SELECTED_N_SAMPLES}"

#############################################
# SWEEP EXECUTION
#############################################

# Construct model path
MODEL_PATH="${BASE_MODEL_PATH}/${STEP}-hf"
MODEL_NAME="${BASE_MODEL_NAME}-${STEP}"

# Check if checkpoint exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "WARNING: Checkpoint not found: ${MODEL_PATH}"
    echo "Skipping..."
    continue
fi

bash evaluate_olmo2_math.sh ${MODEL_PATH} ${MODEL_NAME} ${N_SAMPLES}