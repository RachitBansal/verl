#!/usr/bin/env bash
# Parallel-avg combined SFT+RL training on OLMo2-1B with GSM8K.
# Two independent Adam optimizers (one per loss), per-step update = average of two models.
# Under this mode, sft_config.rl_loss_weight / sft_config.sft_loss_weight are ignored.
#
# RL data:  openmathinstruct2_gsm8k           (rollout-based)
# SFT data: openmathinstruct2_gsm8k            (ground-truth responses, dedupe)
# Both datasets are GSM8K (dedupe); two separate data loaders (same pattern as combined mode).

set -x

# Clean up any existing Ray processes and stale state
echo "Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/ 2>/dev/null || true
sleep 5

# ============================================================================
# Configuration
# ============================================================================
STEP_NUM=${STEP_NUM:-10000}
RL_LR=${RL_LR:-1e-6}
SFT_LR=${SFT_LR:-4e-5}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-50}
# scale_batch_by_rollout_n: when False, SFT mini-batch = RL mini-batch / rollout.n
SCALE_BATCH=${SCALE_BATCH:-True}
if [[ "${SCALE_BATCH}" == "False" ]]; then
    BATCH_TAG="_smbatch"
else
    BATCH_TAG=""
fi

OLMO_CHECKPOINT=/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf

# GPU configuration (auto-detect from SLURM if available)
N_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-2}

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

source ${CONDA_ENV}

# Full sft_optim block as a Hydra/OmegaConf inline dict.
# We pass the entire FSDPOptimizerConfig so the dataclass conversion fills every field.
# Structural fields mirror the defaults in verl/trainer/config/optim/fsdp.yaml.
SFT_OPTIM="{optimizer: AdamW, optimizer_impl: torch.optim, lr: ${SFT_LR}, weight_decay: 0.01, betas: [0.9, 0.999], lr_warmup_steps_ratio: 0.0, lr_warmup_steps: -1, total_training_steps: -1, clip_grad: 1.0, lr_scheduler_type: constant, min_lr_ratio: 0.0, num_cycles: 0.5, grad_clip: null, warmup_style: null, override_optimizer_config: null}"

EXP_NAME="OLMo2-1B_step${STEP_NUM}_parallel_avg_n32_rl${RL_LR}_sft${SFT_LR}${BATCH_TAG}"

# ============================================================================
# Training - Parallel-avg combined SFT+RL mode
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
    sft_config.mode=parallel_avg \
    sft_config.scale_batch_by_rollout_n=${SCALE_BATCH} \
    sft_data.train_files=${SFT_TRAIN_FILE} \
    sft_data.load_ground_truth=True \
    sft_data.response_field_name='generated_solution' \
    actor_rollout_ref.model.path=${OLMO_CHECKPOINT} \
    actor_rollout_ref.actor.optim.lr=${RL_LR} \
    "++actor_rollout_ref.actor.sft_optim=${SFT_OPTIM}" \
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
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl_pretrain' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}/${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=100
