#!/usr/bin/env bash
# Interleaved SFT+RL training on OLMo2-1B with OpenMathInstruct2/GSM8K
# RL data:  openmathinstruct2_gsm8k          (rollout-based)
# SFT data: configurable via SFT_DATA_DIR env var
#   - openmathinstruct2_gsm8k (de-duplicated)
#   - openmathinstruct2_gsm8k_duplicated (duplicated)

set -x

# Clean up any existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop 2>/dev/null || true
sleep 30

# ============================================================================
# Configuration
# ============================================================================
STEP_NUM=${STEP_NUM:-1000}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B"}
OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-${CHECKPOINT_DIR}/step${STEP_NUM}-hf}

# GPU configuration (auto-detect from SLURM if available)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# RL dataset (rollout-based, no pre-written responses needed)
RL_DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
TRAIN_FILE="${RL_DATA_DIR}/train_gsm8k.parquet"
VAL_FILE="${RL_DATA_DIR}/val_gsm8k.parquet"

# SFT dataset (configurable via env var, defaults to duplicated)
SFT_DATA_DIR=${SFT_DATA_DIR:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k_duplicated"}
SFT_TRAIN_FILE="${SFT_DATA_DIR}/train_gsm8k.parquet"

# Output directory for checkpoints
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Experiment name suffix (configurable via env var)
# EXP_TAG: insert between step number and suffix (e.g. "sfted" -> step1000sfted_...)
EXP_TAG=${EXP_TAG:-""}
EXP_SUFFIX=${EXP_SUFFIX:-"interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}"}

# Wandb (optional)
export WANDB_ENTITY="harvardml"

LR_SCALE=${LR_SCALE:-1.0}
NUM_SFT_STEPS=${NUM_SFT_STEPS:-10}
NUM_PPO_STEPS=${NUM_PPO_STEPS:-10}

source ${CONDA_ENV}


# ============================================================================
# Training - Interleaved SFT+RL mode
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    +data.load_ground_truth=False \
    +data.response_field_name='generated_solution' \
    sft_config.enabled=True \
    sft_config.load_ground_truth=True \
    sft_config.mode=interleaved \
    sft_config.num_sft_steps=${NUM_SFT_STEPS} \
    sft_config.num_ppo_steps=${NUM_PPO_STEPS} \
    sft_data.train_files=${SFT_TRAIN_FILE} \
    sft_data.load_ground_truth=True \
    sft_data.response_field_name='generated_solution' \
    actor_rollout_ref.model.path=$OLMO_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=4e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.sft_lr_scale=${LR_SCALE} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_pretrain' \
    trainer.experiment_name="OLMo2-1B_step${STEP_NUM}${EXP_TAG}_${EXP_SUFFIX}" \
    trainer.default_local_dir="${OUTPUT_DIR}/OLMo2-1B_step${STEP_NUM}${EXP_TAG}_${EXP_SUFFIX}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=250 \
    trainer.total_epochs=100
