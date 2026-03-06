#!/usr/bin/env bash
# SFT training with OLMo2-1B on OpenMathInstruct-2 GSM8K subset
# Adapted from your GRPO launch script

set -x

# ============================================================================
# Configuration
# ============================================================================
STEP_NUM=${STEP_NUM:-22000}

OLMO_CHECKPOINT="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf"

# GPU configuration (auto-detect from SLURM if available)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# Dataset
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
TRAIN_FILE="${DATA_DIR}/train_gsm8k.parquet"
VAL_FILE="${DATA_DIR}/val_gsm8k.parquet"

# Output directory for checkpoints
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Wandb (optional)
export WANDB_ENTITY="harvardml"

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# ============================================================================
# Distributed launch
# ============================================================================
# SFT uses torch.distributed directly (not Ray), so launch with torchrun.
# If you're on a single node, this is usually enough.
torchrun --standalone --nproc_per_node=${N_GPUS_PER_NODE} -m verl.trainer.sft_trainer \
    # --- data ---
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=512 \
    data.pad_mode=right \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=4096 \
    data.micro_batch_size_per_gpu=16 \
    # --- model ---
    model.path=${OLMO_CHECKPOINT} \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    # --- engine / strategy ---
    engine.strategy=fsdp \
    # Common FSDP toggles (names may vary slightly in your config)
    engine.fsdp_config.param_offload=False \
    engine.fsdp_config.optimizer_offload=False \
    # --- optimization ---
    optim.lr=1e-6 \
    # --- trainer ---
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_pretrain' \
    trainer.experiment_name="olmo2_1b_step${STEP_NUM}_sft_verl_omigsm8k" \
    trainer.default_local_dir="${OUTPUT_DIR}/olmo2_1b_step${STEP_NUM}_sft_verl_omigsm8k" \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    "$@"