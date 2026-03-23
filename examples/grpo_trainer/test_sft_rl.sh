#!/usr/bin/env bash
# Test SFT-only training (never switch to RL mode)
# This tests that the implementation correctly handles pure supervised fine-tuning
# Author: Claude

set -x

# Clean up any existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop 2>/dev/null || true
sleep 30

# ============================================================================
# Configuration
# ============================================================================
# Model checkpoint
STEP_NUM=${STEP_NUM:-1000}

OLMO_CHECKPOINT=/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf

# GPU configuration (auto-detect from SLURM if available)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset (must have response/answer field for SFT)
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
TRAIN_FILE="${DATA_DIR}/train_gsm8k.parquet"
VAL_FILE="${DATA_DIR}/val_gsm8k.parquet"

# Output directory for checkpoints
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Wandb (optional)
export WANDB_ENTITY="harvardml"
LR_SCALE=${LR_SCALE:-1.0}  # Scale factor for SFT learning rate (e.g., 0.1 to reduce LR during SFT)
num_sft_steps=${num_sft_steps:-10}  # Number of SFT steps to run before switching to PPO (set high for pure SFT)
num_ppo_steps=${num_ppo_steps:-10}  # Number of PPO steps to

source /n/home03/cmohri/venvs/verl_env/bin/activate


# ============================================================================
# SFT Configuration Notes
# ============================================================================
# Loss aggregation modes for SFT:
#   - token-mean: Average loss across all tokens (normalize by token count) [DEFAULT]
#   - token-sum: Sum loss across all tokens without normalization
#                (longer sequences contribute more to gradient)
#   - seq-mean-token-sum: Sum tokens in each sequence, then average across sequences
#
# For typical SFT training with variable-length sequences, use:
#   - token-mean: Good for stable training with balanced gradient contributions
#   - token-sum: Good if you want longer sequences to dominate training
#
# Change by adding: actor_rollout_ref.actor.loss_agg_mode=token-sum

# ============================================================================
# Training - Pure SFT mode
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
    +data.load_ground_truth=True \
    +data.response_field_name='generated_solution' \
    actor_rollout_ref.model.path=$OLMO_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=4e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.sft_lr_scale=${LR_SCALE} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_pretrain' \
    trainer.experiment_name="OLMo2-1B_step${STEP_NUM}_sft_rl_n32_sft_${num_sft_steps}_ppo_${num_ppo_steps}_sanity_separate_data" \
    trainer.default_local_dir="${OUTPUT_DIR}/OLMo2-1B_step${STEP_NUM}_sft_rl_n32_sft_${num_sft_steps}_ppo_${num_ppo_steps}_sanity_separate_data" \
    trainer.experiment_name="OLMo2-1B_step${STEP_NUM}_sft_rl_n32_sft_${num_sft_steps}_ppo_${num_ppo_steps}_sanity_separate_data" \
    trainer.default_local_dir="${OUTPUT_DIR}/OLMo2-1B_step${STEP_NUM}_sft_rl_n32_sft_${num_sft_steps}_ppo_${num_ppo_steps}_sanity_separate_data" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.total_epochs=100 \
    sft_config.enabled=True \
    sft_config.load_ground_truth=True \
    sft_config.num_sft_steps=${num_sft_steps} \
    sft_config.num_ppo_steps=${num_ppo_steps} \
    sft_data.train_files=${TRAIN_FILE} 
