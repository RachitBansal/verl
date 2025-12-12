#!/usr/bin/env bash
# GRPO training with OLMo2-1B on OpenMathInstruct-2 GSM8K subset
# Author: Sunny + Claude (Modified for cmohri env)

set -x

# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================

# Source the venv
source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# Use the env's Python explicitly:
PYBIN=/n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/python

# Export API keys and settings
export WANDB_API_KEY="cdc354a166dcf9cbcd8b1eed1fd2fe89d03f8b90"
export WANDB_ENTITY="harvardml"
export PYTHONNOUSERSITE=1
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1

# ---- Fast local caches (Fixes the dam_lab/Lab permission issue) ----
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache-$USER}
export HF_HOME=${HF_HOME:-/tmp/hf-home-$USER}
mkdir -p "$HF_HOME" || true

# Clean up any existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop 2>/dev/null || true
sleep 10  # Reduced sleep slightly to save time

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

# Model checkpoint (Read-only path is fine here)
STEP_NUM=22000
OLMO_CHECKPOINT="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf"

# GPU configuration (Auto-detect from SLURM, default to 1)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset
# NOTE: Ensure you have read access here. If this is data you generated yourself, 
# you might need to point this to your sham_lab directory instead.
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
TRAIN_FILE="${DATA_DIR}/train_gsm8k.parquet"
VAL_FILE="${DATA_DIR}/val_gsm8k.parquet"

# Output directory for checkpoints
# CRITICAL FIX: Pointing this to your user scratch space to avoid Permission Denied
USER_SCRATCH="/n/netscratch/dam_lab/Everyone/rl_pretrain"
OUTPUT_DIR="${USER_SCRATCH}/rl_pretrain_experiments/experiments"
mkdir -p "$OUTPUT_DIR"

# Reward: partial credit for \boxed{} format (0.1 = 10% reward for formatting)
FORMAT_SCORE=0.1

# ============================================================================
# 3. TRAINING
# ============================================================================

# Using $PYBIN instead of python3 to ensure we use the venv
$PYBIN -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$OLMO_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
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
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward_model.reward_kwargs.format_score=${FORMAT_SCORE} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_pretrain' \
    trainer.experiment_name="olmo2_1b_step${STEP_NUM}_omigsm8k" \
    trainer.default_local_dir="${OUTPUT_DIR}/olmo2_1b_step${STEP_NUM}_omigsm8k" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    "$@"