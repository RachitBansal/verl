#!/bin/bash
#SBATCH --job-name=verl-fixed-rollout
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fixed-rollout-math5k-%j.out
#SBATCH --error=logs/fixed-rollout-math5k-%j.out

# ============================================================================
# Fixed Rollout SLURM Job Script - SANITY CHECK
# Author: Sunny + Claude
#
# SANITY CHECK MODE:
# - Uses MATH examples
# - Fixed n=5 rollouts per sample (configurable via N_ROLLOUTS)
# - Total training epochs configurable via EPOCHS (default: 1)
# - Logs positive rollouts per sample for comparison with adaptive rollouts
# ============================================================================

mkdir -p logs
module purge
module load Mambaforge
module load cuda cudnn

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "============================================================"
echo "FIXED ROLLOUT EXPERIMENT - SANITY CHECK (MATH, n=${N_ROLLOUTS:-5}, epochs=${EPOCHS:-1})"
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

# Configuration
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo-2-0425-1B"}

# Sanity check: Fixed 5 rollouts per sample
N_ROLLOUTS=${N_ROLLOUTS:-5}

# Positive rollout threshold (for logging how many rollouts are correct)
POSITIVE_THRESHOLD=${POSITIVE_THRESHOLD:-0.5}

N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset: 5000 MATH examples for sanity check
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_math"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_math_5000.parquet}"  # 5000 MATH examples for sanity check
VAL_MATH_FILE="${DATA_DIR}/val_math.parquet"

OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments"

FORMAT_SCORE=0.0
export WANDB_ENTITY="harvardml"

# Batch sizes optimized for H100 80GB x 4
# Note: (ppo_mini_batch_size * n_rollouts / n_gpus) must be divisible by ppo_micro_batch_size_per_gpu
# With n=5, n_gpus=4, micro=8: 128*5/4=160, 160%8=0 ✓
# Total rollouts per step: 128 * 5 = 640
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}  # Must be <= TRAIN_BATCH_SIZE
EPOCHS=${EPOCHS:-13}

EXPERIMENT_NAME="fixed_n${N_ROLLOUTS}_math_5000_epochs${EPOCHS}_${SLURM_JOB_ID}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=[${VAL_MATH_FILE}] \
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
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    "$@"

echo "Training completed at $(date)"