#!/usr/bin/env bash
#SBATCH --job-name=verl-grpo-olmo2
#SBATCH --account=kempner_dam_lab        # Change to your account name
#SBATCH --partition=kempner_h100          # Change to your partition name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=32       # Adjust based on your node
#SBATCH --gpus-per-node=4             # Request 4 GPUs
#SBATCH --mem=200G               # Memory per node
#SBATCH --time=00:10:00          # Wall time limit
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ---- Activate your prebuilt env
source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# Use the env's Python explicitly everywhere:
PYBIN=/n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/python
echo "Python: $PYBIN"

# Optional: avoid accidental user-site contamination
export PYTHONNOUSERSITE=1

# ---- Suppress verbose warnings ----
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::ResourceWarning,ignore::FutureWarning"
export RAY_DEDUP_LOGS=1

# ---- Fast local caches (nice to have)
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache-$USER}
export HF_HOME=${HF_HOME:-/tmp/hf-home-$USER}


export HF_HOME=${HF_HOME:-/tmp/hf-home-$USER}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" || true
export HYDRA_FULL_ERROR=1

# ---- Quick preflight: verify essential components ----
$PYBIN - <<'PY'
import torch, flash_attn, vllm
print(f"✓ Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
print(f"✓ FlashAttention: {getattr(flash_attn, '__version__', 'unknown')}")
print(f"✓ vLLM: {vllm.__version__}")
PY

# ---- Set model checkpoint and data paths

CHECKPOINT="/n/netscratch/dam_lab/Everyone/sqin_share/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

DATA_DIR="/n/netscratch/dam_lab/Everyone/sqin_share/gsm8k"

export DISABLE_FLASH_ATTN=0
export USE_EAGER_ATTN=0

export WANDB_MODE=offline

unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING
unset NCCL_DEBUG
unset NCCL_IB_DISABLE
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE

# Run the training script with reduced verbosity
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
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='interleaved_rl' \
    trainer.experiment_name="qwen2.5-0.5b_grpo" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=/n/netscratch/dam_lab/Lab/brachit/verl/outputs \
    "$@"