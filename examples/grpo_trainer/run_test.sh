#!/usr/bin/env bash
#SBATCH --job-name=verl-grpo-test
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

# ==============================================================================
# BITSANDBYTES COMPATIBILITY FIX (2025-10-31)
# ==============================================================================
# ISSUE: Ray workers were failing with "GLIBC_2.34 not found" error when loading
#        bitsandbytes, because prebuilt wheels required GLIBC 2.34 but this
#        system only has GLIBC 2.28.
#
# ROOT CAUSE:
#   1. System GLIBC version: 2.28 (Rocky Linux 8)
#   2. Latest bitsandbytes wheels: require GLIBC 2.34
#   3. Ray workers were falling back to CPU version of bitsandbytes
#
# SOLUTION APPLIED:
#   1. Downgraded to bitsandbytes==0.43.0 (compatible with GLIBC 2.28)
#   2. Set BNB_CUDA_VERSION=123 to use CUDA 12.3 library (compatible with CUDA 12.6)
#   3. Added BNB_CUDA_VERSION to Ray runtime environment in:
#      - verl/trainer/constants_ppo.py (line 33)
#   4. Prevented environment variable filtering in:
#      - verl/trainer/constants_ppo.py (line 52-54)
#   5. Added to ~/.bashrc: export BNB_CUDA_VERSION=123
#
# VERIFICATION:
#   - Check Ray worker logs for: "Loading: libbitsandbytes_cuda123.so"
#   - Warning about BNB_CUDA_VERSION override is EXPECTED and harmless
#
# PACKAGES MODIFIED:
#   - bitsandbytes: 0.45.0 → 0.43.0
#   - triton: 3.3.1 → 3.1.0
# ==============================================================================

DATA_DIR="/n/netscratch/dam_lab/Lab/sqin/tmp_data/gsm8k"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=${DATA_DIR}/train.parquet \
 data.val_files=${DATA_DIR}/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15