#!/usr/bin/env bash
# Non-SLURM version for local/interactive execution
# GRPO training with OLMo2-1B on OpenMathInstruct-1 dataset

set -x

# ============================================================================
# Model Configuration
# ============================================================================
# Use custom local checkpoint instead of HuggingFace hub model
OLMO_CHECKPOINT="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1"

# ============================================================================
# Dataset Configuration - OpenMathInstruct-1
# ============================================================================
# NOTE: You need to download and prepare the OpenMathInstruct-1 dataset first.
# Run: python3 examples/grpo_trainer/prepare_openmath_data.py
DATA_DIR="/n/netscratch/dam_lab/Lab/sqin/tmp_data/openmath"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/validation.parquet"

# ============================================================================
# Environment Configuration
# ============================================================================
export DISABLE_FLASH_ATTN=1
export USE_EAGER_ATTN=0

# Wandb configuration (uncomment and set your key if using wandb)
# export WANDB_API_KEY="your_key_here"
# export WANDB_MODE=offline

# ============================================================================
# Training Configuration
# ============================================================================
python3 -m verl.trainer.main_ppo \
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
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_openmath' \
    trainer.experiment_name='olmo2_1b_openmath_stage1' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    "$@"

