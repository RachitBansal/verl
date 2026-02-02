#!/bin/bash
#SBATCH --job-name=verl-resume
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/resume-%j.out
#SBATCH --error=logs/resume-%j.out

# ============================================================================
# Resume Training Job (crashed or completed)
# 
# Usage:
#   # Resume a crashed job (continues to original epoch target)
#   RESUME_JOB=fixed_n5_hard_epochs10_57675814 sbatch resume_crashed_jobs.sbatch
#
#   # Continue a COMPLETED job for more epochs (e.g., extend from 10 to 20 epochs)
#   RESUME_JOB=fixed_n64_hard_epochs10_57675738 NEW_EPOCHS=20 sbatch resume_crashed_jobs.sbatch
#
#   # Or use the convenience script: ./submit_all_resume_jobs.sh
# ============================================================================

# Change to the verl directory (required for module import)
cd /n/netscratch/dam_lab/Lab/brachit/rollouts/verl

mkdir -p logs
module purge
module load Mambaforge
module load cuda cudnn

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# Must specify which job to resume
if [ -z "$RESUME_JOB" ]; then
    echo "ERROR: RESUME_JOB must be set"
    echo ""
    echo "Usage: RESUME_JOB=<job_name> sbatch resume_crashed_jobs.sbatch"
    exit 1
fi

echo "============================================================"
if [ -n "$NEW_EPOCHS" ]; then
    echo "EXTENDING COMPLETED JOB: ${RESUME_JOB}"
    echo "NEW TARGET EPOCHS: ${NEW_EPOCHS}"
else
    echo "RESUMING JOB: ${RESUME_JOB}"
fi
echo "============================================================"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

set -x

ray stop 2>/dev/null || true
sleep 10

# Configuration
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo-2-0425-1B"}
POSITIVE_THRESHOLD=${POSITIVE_THRESHOLD:-0.5}
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_rollouts"
OUTPUT_DIR="${BASE_DIR}/experiments"
SPLITS_DIR="${BASE_DIR}/splits/1B-0425-64samples-11k"

FORMAT_SCORE=0.0
export WANDB_ENTITY="harvardml"

# Parse job name to extract parameters
# Format: fixed_n{N_ROLLOUTS}_{DATA_TYPE}_epochs{EPOCHS}_{OLD_JOB_ID}
parse_job_name() {
    local job_name=$1
    
    # Extract N_ROLLOUTS
    N_ROLLOUTS=$(echo "$job_name" | sed -n 's/fixed_n\([0-9]*\)_.*/\1/p')
    
    # Extract DATA_TYPE (hard, easy, medium, balanced)
    DATA_TYPE=$(echo "$job_name" | sed -n 's/fixed_n[0-9]*_\([a-z]*\)_epochs.*/\1/p')
    
    # Extract EPOCHS
    EPOCHS=$(echo "$job_name" | sed -n 's/.*_epochs\([0-9]*\)_.*/\1/p')
    
    echo "Parsed: N_ROLLOUTS=$N_ROLLOUTS, DATA_TYPE=$DATA_TYPE, EPOCHS=$EPOCHS"
}

parse_job_name "$RESUME_JOB"

# Validate parsed values
if [ -z "$N_ROLLOUTS" ] || [ -z "$DATA_TYPE" ] || [ -z "$EPOCHS" ]; then
    echo "ERROR: Failed to parse job name: $RESUME_JOB"
    exit 1
fi

# Store original epochs for reference
ORIGINAL_EPOCHS=$EPOCHS

# Allow overriding epochs to continue training beyond original target
# NEW_EPOCHS must be greater than ORIGINAL_EPOCHS to have any effect
if [ -n "$NEW_EPOCHS" ]; then
    if [ "$NEW_EPOCHS" -le "$ORIGINAL_EPOCHS" ]; then
        echo "WARNING: NEW_EPOCHS ($NEW_EPOCHS) <= ORIGINAL_EPOCHS ($ORIGINAL_EPOCHS)"
        echo "         Job will likely finish immediately. Consider setting NEW_EPOCHS > $ORIGINAL_EPOCHS"
    fi
    EPOCHS=$NEW_EPOCHS
    echo "Extending training: $ORIGINAL_EPOCHS -> $NEW_EPOCHS epochs"
fi

# Set data file based on DATA_TYPE
case "$DATA_TYPE" in
    "hard")
        TRAIN_FILE="${SPLITS_DIR}/hard/train.parquet"
        VAL_FILE="${SPLITS_DIR}/hard/val.parquet"
        ;;
    "medium")
        TRAIN_FILE="${SPLITS_DIR}/medium/train.parquet"
        VAL_FILE="${SPLITS_DIR}/medium/val.parquet"
        ;;
    "easy")
        TRAIN_FILE="${SPLITS_DIR}/easy/train.parquet"
        VAL_FILE="${SPLITS_DIR}/easy/val.parquet"
        ;;
    "balanced")
        TRAIN_FILE="${SPLITS_DIR}/balanced/train.parquet"
        VAL_FILE="${SPLITS_DIR}/balanced/val.parquet"
        ;;
    *)
        echo "ERROR: Unknown DATA_TYPE: $DATA_TYPE"
        exit 1
        ;;
esac

# Validation files
VAL_GSM_FILE="${BASE_DIR}/data/openmathinstruct2/val_gsm8k.parquet"
VAL_MATH_FILE="${BASE_DIR}/data/openmathinstruct2/val_math.parquet"

# The original experiment directory (where checkpoints are)
EXPERIMENT_DIR="${OUTPUT_DIR}/${RESUME_JOB}"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Find latest checkpoint
LATEST_STEP=$(ls "$EXPERIMENT_DIR" | grep "global_step_" | sed 's/global_step_//' | sort -n | tail -1)
echo "Latest checkpoint: global_step_${LATEST_STEP}"

# Use the original experiment name to maintain continuity in wandb
EXPERIMENT_NAME="${RESUME_JOB}"

# Batch sizes
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

echo ""
echo "Configuration:"
echo "  N_ROLLOUTS: $N_ROLLOUTS"
echo "  DATA_TYPE: $DATA_TYPE"
echo "  ORIGINAL_EPOCHS: $ORIGINAL_EPOCHS"
echo "  TARGET_EPOCHS: $EPOCHS"
echo "  TRAIN_FILE: $TRAIN_FILE"
echo "  EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "  RESUMING FROM: step $LATEST_STEP"
if [ -n "$NEW_EPOCHS" ]; then
    echo "  MODE: Extending completed job ($ORIGINAL_EPOCHS -> $EPOCHS epochs)"
else
    echo "  MODE: Resuming crashed job (target: $EPOCHS epochs)"
fi
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=[${VAL_GSM_FILE},${VAL_FILE}] \
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
    trainer.default_local_dir="${EXPERIMENT_DIR}" \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    "$@"

echo "Training completed at $(date)"
