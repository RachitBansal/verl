#!/usr/bin/env bash
#SBATCH --job-name=verl-grpo-olmo2-openmath
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner          # Change to your partition name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=32       # Adjust based on your node
#SBATCH --gpus-per-node=4             # Request 4 GPUs
#SBATCH --mem=200G               # Memory per node
#SBATCH --time=48:00:00          # Wall time limit (increased for large dataset)
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -x

echo "Starting training on $(hostname) with $SLURM_NTASKS_PER_NODE tasks and $SLURM_GPUS_PER_NODE GPUs..."

# ============================================================================
# Model Configuration
# ============================================================================
# Use custom local checkpoint instead of HuggingFace hub model
OLMO_CHECKPOINT="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1"

# ============================================================================
# Dataset Configuration - OpenMathInstruct-1
# ============================================================================
# NOTE: You need to download and prepare the OpenMathInstruct-1 dataset first.
# The dataset should be in parquet format with columns matching the expected format.
#
# To download and prepare the dataset, you can use:
#   from datasets import load_dataset
#   dataset = load_dataset("nvidia/OpenMathInstruct-1")
#   dataset['train'].to_parquet("train.parquet")
#   dataset['validation'].to_parquet("validation.parquet")
#
# Expected format: Each row should have 'question' field (prompt) and optionally
# 'expected_answer' or 'predicted_answer' for evaluation.

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
# export WANDB_MODE=offline  # Save logs locally without uploading

# ============================================================================
# Training Configuration
# ============================================================================
# OpenMathInstruct-1 has longer solutions with code blocks, so we need:
# - Longer max_response_length for code generation
# - Appropriate batch sizes for the larger dataset (1.8M samples)
# - Math-specific training parameters

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

echo "Training completed!"

