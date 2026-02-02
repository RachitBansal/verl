#!/bin/bash
#SBATCH --job-name=verl-fixed-rollout
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fixed-rollout-%j.out
#SBATCH --error=logs/fixed-rollout-%j.out
# ============================================================================
# Fixed Rollout SLURM Job Script
# Author: Sunny + Claude
#
# Train on difficulty-based splits from OpenMathInstruct-2 (64 samples/example)
# Each split has ~10k train, ~1k val examples:
#   - hard: accuracy 0-3.125% (0-2/64 correct)
#   - medium: accuracy 3.125%-12.5% (2-8/64 correct)
#   - easy: accuracy 25%+ (16+/64 correct)
#   - balanced: equal mix (~3.3k each)
# ============================================================================

mkdir -p logs
module purge
module load Mambaforge
module load cuda cudnn

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "============================================================"
echo "FIXED ROLLOUT EXPERIMENT (n=${N_ROLLOUTS:-5}, DATA_TYPE=${DATA_TYPE:-balanced})"
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

# Data type: hard, medium, easy, balanced
DATA_TYPE=${DATA_TYPE:-hard}

# Configuration
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo-2-0425-1B"}

# Fixed rollouts per sample
N_ROLLOUTS=${N_ROLLOUTS:-5}

# Positive rollout threshold (for logging how many rollouts are correct)
POSITIVE_THRESHOLD=${POSITIVE_THRESHOLD:-0.5}

N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset configuration
BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_rollouts"
OUTPUT_DIR="${BASE_DIR}/experiments"

# Difficulty splits directory (64 samples, 11k examples per split)
SPLITS_DIR="${BASE_DIR}/splits/1B-0425-64samples-11k"

FORMAT_SCORE=0.0
export WANDB_ENTITY="harvardml"

# Batch sizes optimized for H100 80GB x 4
# Note: (ppo_mini_batch_size * n_rollouts / n_gpus) must be divisible by ppo_micro_batch_size_per_gpu
# With n=5, n_gpus=4, micro=8: 128*5/4=160, 160%8=0 ✓
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}

# Validation files
VAL_GSM_FILE="${BASE_DIR}/data/openmathinstruct2/val_gsm8k.parquet"
VAL_MATH_FILE="${BASE_DIR}/data/openmathinstruct2/val_math.parquet"

# Set data directory and epochs based on DATA_TYPE
# Each split has ~10k train examples -> ~78 steps/epoch with batch=128
case "$DATA_TYPE" in
    "hard")
        echo "Using HARD split (accuracy 0-3.125%)"
        DATA_DIR="${SPLITS_DIR}/hard"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_SPLIT_FILE="${DATA_DIR}/val.parquet"
        EPOCHS=${EPOCHS:-5}  # ~1000 steps
        ;;
    "medium")
        echo "Using MEDIUM split (accuracy 3.125%-12.5%)"
        DATA_DIR="${SPLITS_DIR}/medium"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_SPLIT_FILE="${DATA_DIR}/val.parquet"
        EPOCHS=${EPOCHS:-5}
        ;;
    "easy")
        echo "Using EASY split (accuracy 25%+)"
        DATA_DIR="${SPLITS_DIR}/easy"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_SPLIT_FILE="${DATA_DIR}/val.parquet"
        EPOCHS=${EPOCHS:-5}
        ;;
    "balanced")
        echo "Using BALANCED split (~3.3k hard, ~3.3k medium, ~3.3k easy)"
        DATA_DIR="${SPLITS_DIR}/balanced"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_SPLIT_FILE="${DATA_DIR}/val.parquet"
        EPOCHS=${EPOCHS:-5}
        ;;
    *)
        echo "ERROR: Unknown DATA_TYPE '${DATA_TYPE}'"
        echo "Valid types: hard, medium, easy, balanced"
        exit 1
        ;;
esac

echo "Train file: ${TRAIN_FILE}"
echo "Epochs: ${EPOCHS}"

EXPERIMENT_NAME="fixed_n${N_ROLLOUTS}_${DATA_TYPE}_epochs${EPOCHS}_${SLURM_JOB_ID}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=[${VAL_GSM_FILE},${VAL_SPLIT_FILE}] \
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
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward_model.reward_kwargs.format_score=${FORMAT_SCORE} \
    +positive_rollout_threshold=${POSITIVE_THRESHOLD} \
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
    "$@"

echo "Training completed at $(date)"
