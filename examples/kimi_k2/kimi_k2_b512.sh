#!/usr/bin/env bash
#SBATCH --job-name=kimi_k2
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -xeuo pipefail

# Twin submissions: the same cell submitted to several partitions under one --job-name; whichever starts
# first cancels the still-pending twins (CANCEL_TWINS=1 at submit; the name must be unique to the cell).
if [ -n "${CANCEL_TWINS:-}" ]; then
    scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true
fi

####################
# Kimi-K2 squared-regression objective on the CBS GRPO spine
# (Qwen2.5-Math-1.5B-Instruct, DAPO-Math-17k, AIME 1983-2024 val; same data/model/lengths/eval as
#  examples/grpo_trainer/cbs_experiments/on_policy_brachit.sh so the grpo_on_policy_cbs runs are controls).
#
# Usage (from repo root, env vars select the cell):
#   BSZ=512 N=16 LR=1e-5 TAU=0.01 MINI=128 EPOCHS=1 sbatch examples/kimi_k2/kimi_k2_b512.sh
#   LOSS_MODE=vanilla MINI=128 ...   -> GRPO (PPO-clip) control with the same mini-batching
# Knobs:
#   MINI    prompts per optimizer step (MINI=BSZ -> one step per rollout batch == Dr. GRPO)
#   EPOCHS  passes over the rollout batch (ppo_epochs)
#   TAU     kimi_tau: target log-ratio per sequence = (r - rbar)/TAU nats
#   LEVEL   seq_sum (published) | seq_mean
#   FORM    squared (published) | gradient
#   KL_COEF KL-to-reference loss coefficient (0 = published K2; the GRPO controls used 0.001)
#   STD_ADV True -> GRPO std-normalised advantages as the target (controls use True; published K2 is False)
#   CLIP_LOW/CLIP_HIGH  PPO clip range for LOSS_MODE=vanilla (0.2/0.2 = cbs GRPO; 0.2/0.4 = DAPO clip-higher)
#   HOLDOUT rows held out of the update for the held-out regression loss (per DP rank)
####################

BSZ=${BSZ:-512}
N=${N:-16}
LR=${LR:-1e-5}
TAU=${TAU:-0.01}
MINI=${MINI:-128}
EPOCHS=${EPOCHS:-1}
LEVEL=${LEVEL:-seq_sum}
FORM=${FORM:-squared}
KL_COEF=${KL_COEF:-0}
STD_ADV=${STD_ADV:-False}
LOSS_MODE=${LOSS_MODE:-kimi_k2}
CLIP_LOW=${CLIP_LOW:-0.2}     # PPO clip range (vanilla/GRPO only); asymmetric clip_high=0.4 is the DAPO "clip-higher"
CLIP_HIGH=${CLIP_HIGH:-0.2}
PROBE=${PROBE:-64}
# Held-out regression rows. HOLDOUT_PROMPTS>0: that many extra prompts are generated every step and held
# out of the update (fresh prompts; the training batch/mini-batches are unchanged; +HOLDOUT_PROMPTS/BSZ
# rollouts). HOLDOUT (legacy): rows per DP rank carved out of the training batch instead; off when
# HOLDOUT_PROMPTS>0.
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-128}
if [ "${HOLDOUT_PROMPTS}" -gt 0 ]; then HOLDOUT=${HOLDOUT:-0}; else HOLDOUT=${HOLDOUT:-64}; fi
TRACE=${TRACE:-True}     # kimi_probe_per_step: probe/held-out stats after every inner step
LOGRATIO_CLIP=${LOGRATIO_CLIP:-null}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-100}
TOTAL_STEPS=${TOTAL_STEPS:-400}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}   # dataloader epochs; must cover TOTAL_STEPS (17k prompts / BSZ per epoch)
TAG=${TAG:-}

project_name=${PROJECT:-"kimi_k2_cbs"}
if [ "${LOSS_MODE}" = "kimi_k2" ]; then
    experiment_name="k2_n${N}_bsz${BSZ}_tau${TAU}_${LEVEL}_${FORM}_mini${MINI}_ep${EPOCHS}_lr${LR}_kl${KL_COEF}"
else
    experiment_name="${LOSS_MODE}_n${N}_bsz${BSZ}_mini${MINI}_ep${EPOCHS}_lr${LR}_kl${KL_COEF}_std${STD_ADV}_clip${CLIP_LOW}-${CLIP_HIGH}"
fi
if [ -n "${TAG}" ]; then experiment_name="${experiment_name}_${TAG}"; fi

source /n/holylabs/dam_lab/Lab/brachit/envs/bin/activate
# verl is not pip-installed in this env; import from this source tree (branch kimi-k2).
# pyextras: wandb deps (platformdirs, gitpython) missing from the lab env, which is admin-writable only.
# VERL_ROOT / PYEXTRAS: override when running from a copy of the tree (e.g. the shared snapshot for collaborators).
VERL_ROOT=${VERL_ROOT:-/n/home08/brachit/coverage/verl}
PYEXTRAS=${PYEXTRAS:-$(dirname "${VERL_ROOT}")/pyextras}
# clones without a sibling pyextras/ (wandb deps missing from the lab env): use the shared copy
[ -d "${PYEXTRAS}" ] || PYEXTRAS=/n/netscratch/dam_lab/Everyone/brachit/share/kimi_k2_hp128/pyextras
export PYTHONPATH=${VERL_ROOT}:${PYEXTRAS}:${PYTHONPATH:-}
cd "${VERL_ROOT}"

OUTPUT_DIR=${OUTPUT_DIR:-"/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_runs"}

# Per-job compile/cache isolation (concurrent compiles on a node race and hit ESTALE).
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=/tmp/hf_${SLURM_JOB_ID}/datasets
mkdir -p "${HF_DATASETS_CACHE}"
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}"
export WANDB_ENTITY=${WANDB_ENTITY:-harvardml}
export RAY_TMPDIR=/tmp/ray_brachit_${SLURM_JOB_ID}
mkdir -p "${RAY_TMPDIR}"
ray stop --force 2>/dev/null || true
sleep 5

use_kl_loss=False
if [ "${KL_COEF}" != "0" ]; then use_kl_loss=True; fi

gpu_memory_utilization=0.7
gen_tp=1
max_prompt_length=1024
max_response_length=3072
data_truncation='left'

model_path=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct
train_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/dapo.parquet
test_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/aime1983_2024.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=${STD_ADV} \
    algorithm.use_kl_in_reward=False \
    data.train_files=${train_data} \
    data.val_files=${test_data} \
    data.train_batch_size=${BSZ} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=${data_truncation} \
    data.kimi_holdout_prompts=${HOLDOUT_PROMPTS} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-600} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI} \
    actor_rollout_ref.actor.ppo_epochs=${EPOCHS} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
    actor_rollout_ref.actor.policy_loss.kimi_tau=${TAU} \
    actor_rollout_ref.actor.policy_loss.kimi_target_level=${LEVEL} \
    actor_rollout_ref.actor.policy_loss.kimi_form=${FORM} \
    actor_rollout_ref.actor.policy_loss.kimi_logratio_clip=${LOGRATIO_CLIP} \
    actor_rollout_ref.actor.policy_loss.kimi_probe_num_seqs=${PROBE} \
    actor_rollout_ref.actor.policy_loss.kimi_holdout_num_seqs=${HOLDOUT} \
    actor_rollout_ref.actor.policy_loss.kimi_probe_per_step=${TRACE} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir="${OUTPUT_DIR}/${experiment_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.max_actor_ckpt_to_keep=${CKPT_KEEP:-2} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.total_epochs=${TOTAL_EPOCHS}
