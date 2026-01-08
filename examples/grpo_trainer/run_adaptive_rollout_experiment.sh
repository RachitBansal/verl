#!/usr/bin/env bash
# Adaptive Rollout Experiment: Variable rollouts (up to 128) with early stopping
# This experiment generates rollouts until a positive (correct) sample is found
# Author: Sunny + Claude

set -x

# Clean up any existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop 2>/dev/null || true
sleep 30

# ============================================================================
# Configuration
# ============================================================================
# Model checkpoint
STEP_NUM=${STEP_NUM:-22000}
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf"}

# Adaptive rollout settings
MAX_N=${MAX_N:-128}                    # Maximum rollouts per sample
ROLLOUTS_PER_BATCH=${ROLLOUTS_PER_BATCH:-8}  # Rollouts generated per iteration
POSITIVE_THRESHOLD=${POSITIVE_THRESHOLD:-0.5}  # Reward threshold for "positive"
MIN_ROLLOUTS=${MIN_ROLLOUTS:-2}        # Minimum rollouts per sample

# GPU configuration
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_GSM_FILE="${DATA_DIR}/val_gsm8k.parquet"
VAL_MATH_FILE="${DATA_DIR}/val_math.parquet"

# Output directory
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"
EXPERIMENT_NAME="olmo2_1b_step${STEP_NUM}_adaptive_n${MAX_N}"

# Reward settings
FORMAT_SCORE=0.1

# Wandb
export WANDB_ENTITY="harvardml"

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# ============================================================================
# Training with Adaptive Rollouts
# ============================================================================
# NOTE: With adaptive rollouts, we use a smaller train_batch_size (number of prompts)
# since we generate many rollouts per prompt (up to max_n=128).
# Example: 32 prompts × avg 20 rollouts = 640 total samples for training
#
# Important: ppo_mini_batch_size must be <= train_batch_size for validation,
# but the actual mini-batching happens on the expanded rollout batch.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}  # Number of prompts per step
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}  # Must be <= TRAIN_BATCH_SIZE

export CUDA_VISIBLE_DEVICES=0

python3 -m verl.trainer.main_ppo_adaptive \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=[${VAL_GSM_FILE},${VAL_MATH_FILE}] \
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
    trainer.project_name='rl_pretrain_adaptive' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@"

