#!/bin/bash
#SBATCH --job-name=verl-adaptive-rollout
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/adaptive-rollout-math5k-%j.out
#SBATCH --error=logs/adaptive-rollout-math5k-%j.out

# ============================================================================
# Adaptive Rollout SLURM Job Script - SANITY CHECK
# Author: Sunny + Claude
#
# SANITY CHECK MODE:
# - Uses 5000 MATH examples only
# - Adaptive rollouts (up to max_n) with early stopping on positive
# - Logs positive rollouts per sample for comparison with fixed rollouts
# ============================================================================

mkdir -p logs
module purge
module load Mambaforge
module load cuda cudnn

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "============================================================"
echo "ADAPTIVE ROLLOUT EXPERIMENT - SANITY CHECK (5000 MATH)"
echo "============================================================"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

set -x

ray stop 2>/dev/null || true
sleep 10

# Configuration
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo-2-0425-1B"}

# Adaptive rollout settings - optimized for parallel generation
# The adaptive rollout now generates large batches in parallel via vLLM,
# then simulates early stopping in post-processing. This is much faster
# than the original sequential approach.
#
# With ROLLOUTS_PER_BATCH=64 and MAX_N=128, we only need 2 vLLM calls max
# instead of 32 calls with ROLLOUTS_PER_BATCH=4.
MAX_N=${MAX_N:-128}                     # Maximum rollouts per sample
ROLLOUTS_PER_BATCH=${ROLLOUTS_PER_BATCH:-64}  # Rollouts generated per iteration (larger = faster, leverages vLLM parallelism)
POSITIVE_THRESHOLD=${POSITIVE_THRESHOLD:-0.5}  # Reward threshold for "positive"
MIN_ROLLOUTS=${MIN_ROLLOUTS:-4}        # Minimum rollouts per sample

# Per-prompt normalization: ensures each prompt contributes equally to loss
# regardless of how many rollouts it took. Set to true to enable.
USE_PROMPT_NORMALIZATION=${USE_PROMPT_NORMALIZATION:-true}

N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_rollouts"
OUTPUT_DIR="${BASE_DIR}/experiments"

export WANDB_ENTITY="harvardml"

# Batch sizes optimized for H100 80GB x 4
# Note: (ppo_mini_batch_size * n / n_gpus) must be divisible by ppo_micro_batch_size_per_gpu
# With n=1, n_gpus=4, micro=4: 16*1/4=4, 4%4=0 ✓
# Actual rollouts per step: 16 prompts × avg_rollouts (4 to MAX_N)
# Conservative: 16 × 40 avg = ~640 rollouts (similar to fixed)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}  # Number of prompts per step
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}  # Must be <= TRAIN_BATCH_SIZE
EPOCHS=${EPOCHS:-1}

DATASET="gsm8k"
DATA_TYPE="easy"  # usual, balanced, hard, medium, easy

if [ "$DATA_TYPE" = "usual" ]; then
    echo "Usual data type"
    DATA_DIR="${BASE_DIR}/data/openmathinstruct2_${DATASET}"
    TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_${DATASET}.parquet}"
    VAL_FILE="${DATA_DIR}/val_${DATASET}.parquet"
elif [ "$DATA_TYPE" = "balanced" ]; then
    echo "Balanced data type"
    DATA_DIR="${BASE_DIR}/splits/1B-0425/${DATASET}/balanced"
    TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
    VAL_FILE="${BASE_DIR}/data/openmathinstruct2_${DATASET}/val_${DATASET}.parquet"
    EPOCHS=13
elif [ "$DATA_TYPE" = "hard" ]; then
    echo "Hard data type"
    DATA_DIR="${BASE_DIR}/splits/1B-0425/${DATASET}/balanced"
    TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/hard_only.parquet}"
    VAL_FILE="${BASE_DIR}/data/openmathinstruct2_${DATASET}/val_${DATASET}.parquet"
    EPOCHS=36
elif [ "$DATA_TYPE" = "medium" ]; then
    echo "Medium data type"
    DATA_DIR="${BASE_DIR}/splits/1B-0425/${DATASET}/balanced"
    TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/medium_only.parquet}"
    VAL_FILE="${BASE_DIR}/data/openmathinstruct2_${DATASET}/val_${DATASET}.parquet"
    EPOCHS=36
elif [ "$DATA_TYPE" = "easy" ]; then
    echo "Easy data type"
    DATA_DIR="${BASE_DIR}/splits/1B-0425/${DATASET}/balanced"
    TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/easy_only.parquet}"
    VAL_FILE="${BASE_DIR}/data/openmathinstruct2_${DATASET}/val_${DATASET}.parquet"
    EPOCHS=36
fi

EXPERIMENT_NAME="adaptive_max${MAX_N}_${DATASET}_${DATA_TYPE}_epochs${EPOCHS}_${SLURM_JOB_ID}"
if $USE_PROMPT_NORMALIZATION; then
    EXPERIMENT_NAME="adaptive_norm_max${MAX_N}_${DATASET}_${DATA_TYPE}_epochs${EPOCHS}_${SLURM_JOB_ID}"
fi
FORMAT_SCORE=0.0

# Build loss_agg_mode argument conditionally
if [ "$USE_PROMPT_NORMALIZATION" = "true" ]; then
    LOSS_AGG_MODE_ARG="actor_rollout_ref.actor.loss_agg_mode=prompt-mean-token-sum"
    echo "Per-prompt normalization ENABLED (loss_agg_mode=prompt-mean-token-sum)"
else
    LOSS_AGG_MODE_ARG=""
    echo "Per-prompt normalization DISABLED (using default loss_agg_mode)"
fi

python3 -m verl.trainer.main_ppo_adaptive \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=[${VAL_FILE}] \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$OLMO_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward_model.reward_kwargs.format_score=${FORMAT_SCORE} \
    +adaptive_rollout.enable=True \
    +adaptive_rollout.max_n=${MAX_N} \
    +adaptive_rollout.rollouts_per_batch=${ROLLOUTS_PER_BATCH} \
    +adaptive_rollout.positive_threshold=${POSITIVE_THRESHOLD} \
    +adaptive_rollout.min_rollouts=${MIN_ROLLOUTS} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_rollouts' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    ${LOSS_AGG_MODE_ARG:+$LOSS_AGG_MODE_ARG} \
    "$@"

echo "Training completed at $(date)"