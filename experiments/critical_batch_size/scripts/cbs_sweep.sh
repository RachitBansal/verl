#!/usr/bin/env bash
# Critical Batch Size sweep: parameterized GRPO training for CBS measurement.
# Usage: bash cbs_sweep.sh --n_prompts 128 --n_rollouts 8 [--phase p1] [--model_path ...] [--extra_args ...]

set -x
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
N_PROMPTS=128
N_ROLLOUTS=8
PHASE="p1"
MODEL_PATH="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo-2-0425-1B"
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_GSM_FILE="${DATA_DIR}/val_gsm8k.parquet"
VAL_MATH_FILE="${DATA_DIR}/val_math.parquet"
OUTPUT_BASE="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments/cbs"
LR="1e-6"
FORMAT_SCORE="0.1"
N_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
TOTAL_EPOCHS=5
TEST_FREQ=25
SAVE_FREQ=100
SEED=42
EXTRA_ARGS=""

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_prompts)     N_PROMPTS="$2"; shift 2 ;;
        --n_rollouts)    N_ROLLOUTS="$2"; shift 2 ;;
        --phase)         PHASE="$2"; shift 2 ;;
        --model_path)    MODEL_PATH="$2"; shift 2 ;;
        --data_dir)      DATA_DIR="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --total_epochs)  TOTAL_EPOCHS="$2"; shift 2 ;;
        --test_freq)     TEST_FREQ="$2"; shift 2 ;;
        --save_freq)     SAVE_FREQ="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --n_gpus)        N_GPUS_PER_NODE="$2"; shift 2 ;;
        --extra_args)    EXTRA_ARGS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Recompute data paths in case --data_dir was overridden.
# Auto-detect available train/val files in the data directory.
if [ -f "${DATA_DIR}/train.parquet" ]; then
    TRAIN_FILE="${DATA_DIR}/train.parquet"
elif [ -f "${DATA_DIR}/train_math_100000.parquet" ]; then
    TRAIN_FILE="${DATA_DIR}/train_math_100000.parquet"
else
    echo "ERROR: No train parquet found in ${DATA_DIR}"; exit 1
fi

VAL_FILES=""
[ -f "${DATA_DIR}/val_gsm8k.parquet" ] && VAL_FILES="${VAL_FILES:+${VAL_FILES},}${DATA_DIR}/val_gsm8k.parquet"
[ -f "${DATA_DIR}/val_math.parquet" ] && VAL_FILES="${VAL_FILES:+${VAL_FILES},}${DATA_DIR}/val_math.parquet"
if [ -z "$VAL_FILES" ]; then
    echo "ERROR: No val parquet found in ${DATA_DIR}"; exit 1
fi

# ============================================================================
# Derived configuration
# ============================================================================
TOTAL_BATCH=$((N_PROMPTS * N_ROLLOUTS))

# mini-batch size: use the total batch or 256, whichever is smaller,
# to keep gradient accumulation steps reasonable
if [ "$TOTAL_BATCH" -le 256 ]; then
    MINI_BATCH_SIZE=$TOTAL_BATCH
else
    MINI_BATCH_SIZE=256
fi

# micro-batch (per-GPU forward pass) size: fit within GPU memory
# For 1B model with 2048 response length on H100, 16 per GPU is safe
MICRO_BATCH_PER_GPU=16

EXPERIMENT_NAME="cbs_${PHASE}_np${N_PROMPTS}_nr${N_ROLLOUTS}_lr${LR}"
OUTPUT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"

echo "============================================================"
echo "CBS Sweep: ${EXPERIMENT_NAME}"
echo "  n_prompts=${N_PROMPTS}, n_rollouts=${N_ROLLOUTS}"
echo "  total_batch=${TOTAL_BATCH}, mini_batch=${MINI_BATCH_SIZE}"
echo "  model=${MODEL_PATH}"
echo "  output=${OUTPUT_DIR}"
echo "============================================================"

# ============================================================================
# Environment
# ============================================================================
ray stop 2>/dev/null || true
sleep 10

export WANDB_ENTITY="harvardml"

# ============================================================================
# Training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="[${VAL_FILES}]" \
    data.train_batch_size=${N_PROMPTS} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.seed=${SEED} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward_model.reward_kwargs.format_score=${FORMAT_SCORE} \
    trainer.critic_warmup=0 \
    +trainer.grad_noise_measure_freq=25 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='cbs_rlvr' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    ${EXTRA_ARGS}
