#!/usr/bin/env bash
# Curriculum SFT→RL training on OLMo2-1B with OpenMathInstruct2/GSM8K
# Starts with many SFT steps and few RL steps, gradually shifting toward pure RL.
# Each round: num_ppo_steps += RL_INCREMENT, num_sft_steps -= SFT_DECREMENT (floor 0).
# Once num_sft_steps reaches 0, all subsequent steps are pure RL.
#
# RL data:  openmathinstruct2_gsm8k          (rollout-based)
# SFT data: openmathinstruct2_gsm8k_duplicated (ground-truth responses)

set -x

# Clean up any existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop 2>/dev/null || true
sleep 30

# ============================================================================
# Configuration
# ============================================================================
STEP_NUM=${STEP_NUM:-1000}

OLMO_CHECKPOINT=/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf

# GPU configuration (auto-detect from SLURM if available)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

# RL dataset (rollout-based, no pre-written responses needed)
RL_DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
TRAIN_FILE="${RL_DATA_DIR}/train_gsm8k.parquet"
VAL_FILE="${RL_DATA_DIR}/val_gsm8k.parquet"

# SFT dataset (pre-written ground-truth responses) — de-duplicated
SFT_DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
SFT_TRAIN_FILE="${SFT_DATA_DIR}/train_gsm8k.parquet"

# Output directory for checkpoints
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Wandb (optional)
export WANDB_ENTITY="harvardml"

LR_SCALE=${LR_SCALE:-1.0}

# Curriculum schedule: begin SFT-heavy, shift to RL-only over time.
# Round k: num_sft_steps = max(0, 200 - k), num_ppo_steps = 1 + k
# After 200 rounds, num_sft_steps = 0 → every step is pure RL.
NUM_SFT_STEPS=${NUM_SFT_STEPS:-200}
NUM_PPO_STEPS=${NUM_PPO_STEPS:-1}
RL_INCREMENT=${RL_INCREMENT:-1}
SFT_DECREMENT=${SFT_DECREMENT:-1}

source ${CONDA_ENV:-/n/home03/cmohri/venvs/verl_env/bin/activate}


# ============================================================================
# Training - Curriculum SFT→RL mode
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
    +sft_config.rl_increment=${RL_INCREMENT} \
    +sft_config.sft_decrement=${SFT_DECREMENT} \
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
    trainer.experiment_name="OLMo2-1B_step${STEP_NUM}_curriculum_sft${NUM_SFT_STEPS}_ppo${NUM_PPO_STEPS}_rlinc${RL_INCREMENT}_sftdec${SFT_DECREMENT}" \
    trainer.default_local_dir="${OUTPUT_DIR}/OLMo2-1B_step${STEP_NUM}_curriculum_sft${NUM_SFT_STEPS}_ppo${NUM_PPO_STEPS}_rlinc${RL_INCREMENT}_sftdec${SFT_DECREMENT}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.total_epochs=100
