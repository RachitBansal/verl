#!/usr/bin/env bash
#SBATCH --job-name=verl-grpo-olmo2
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner          # Change to your partition name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=32       # Adjust based on your node
#SBATCH --gpus-per-node=4             # Request 4 GPUs
#SBATCH --mem=200G               # Memory per node
#SBATCH --time=24:00:00          # Wall time limit
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -x

echo "Starting training on $(hostname) with $SLURM_NTASKS_PER_NODE tasks and $SLURM_GPUS_PER_NODE GPUs..."

CHECKPOINT="/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

DATA_DIR="/n/netscratch/dam_lab/Lab/sqin/tmp_data/gsm8k"

export DISABLE_FLASH_ATTN=1
export USE_EAGER_ATTN=0

export WANDB_API_KEY="e1685e2a871e986f19e07c24c1d4dd878b7fde35"
# export WANDB_MODE=offline

# Run the training script
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=${CHECKPOINT} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    trainer.logger='["console"]' \
    trainer.project_name='interleaved_rl' \
    trainer.experiment_name="qwen2.5-0.5b_grpo" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "$@"

echo "Training completed!"

