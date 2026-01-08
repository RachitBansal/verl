#!/usr/bin/env bash
# GRPO training with OLMo2-1B on OpenMathInstruct-2 GSM8K subset
# Author: Sunny + Claude

set -x

# Clean up any existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop 2>/dev/null || true
sleep 30

# ============================================================================
# Configuration
# ============================================================================
# Model checkpoint (can be overridden by exporting STEP_NUM or OLMO_CHECKPOINT)
STEP_NUM=${STEP_NUM:-22000}
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf"}
N_ROLLOUTS=${N_ROLLOUTS:-32}

# GPU configuration (auto-detect from SLURM if available)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset (run: python3 examples/data_preprocess/openmathinstruct2.py)
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_GSM_FILE="${DATA_DIR}/val_gsm8k.parquet"
VAL_MATH_FILE="${DATA_DIR}/val_math.parquet"

# Output directory for checkpoints
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Reward: partial credit for \boxed{} format (0.1 = 10% reward for formatting)
FORMAT_SCORE=0.1

# Wandb (optional)
# export WANDB_API_KEY="your_key_here"
export WANDB_ENTITY="harvardml"  # Ensure all team members log to same entity

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# ============================================================================
# Training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=[${VAL_GSM_FILE},${VAL_MATH_FILE}] \
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
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward_model.reward_kwargs.format_score=${FORMAT_SCORE} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_pretrain' \
    trainer.experiment_name="olmo2_1b_step${STEP_NUM}_omi_n64" \
    trainer.default_local_dir="${OUTPUT_DIR}/olmo2_1b_step${STEP_NUM}_omi_n64" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    "$@"

