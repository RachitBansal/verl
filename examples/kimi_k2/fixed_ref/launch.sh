#!/usr/bin/env bash
#SBATCH --job-name=k2fix
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -xeuo pipefail
if [ -n "${CANCEL_TWINS:-}" ]; then
    scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true
fi

####################
# Fixed-pi_ref-dataset Kimi-K2 study (coverage/Phase2a_Fixed_PiRef_Dataset_Plan.md).
# One bank of B x N reference rollouts (+ held-out fresh prompts / fresh samples), built once at the reference
# weights (MODE=build) and then trained on with the K2 loss, anchor frozen at the reference log-probs, one
# full-batch Adam step per step (ppo_mini_batch_size = B).  Same model/data/lengths/eval as kimi_k2_b512.sh.
#
#   MODE=build sbatch examples/kimi_k2/fixed_ref/launch.sh            # ~30 min: generate + score + log-probs, save bank
#   TAU=0.03 LR=1e-5 sbatch examples/kimi_k2/fixed_ref/launch.sh     # train on the bank
# Knobs: TAU LR ANCHOR(ref|current) BASELINE(mean|tau_logz) STEPS TEST_FREQ SAVE_FREQ HF_SAVE_FREQ HF_DENSE VAL_N BANK_DIR PROMPT_DIR TAG
# Master bank (README.md): MODE=master_build JOB_INDEX=j NUM_JOBS=J [N_MASTER=64 CHUNK=512 MASTER_DIR=...] builds shards;
#   SELECTION=<dir from prepare_master_selection.py> TAU=... LR=... trains on that subset (BSZ, N taken from it)
####################
MODE=${MODE:-train}
BSZ=${BSZ:-4096}
N=${N:-16}
LR=${LR:-1e-5}
TAU=${TAU:-0.03}
ANCHOR=${ANCHOR:-ref}
BASELINE=${BASELINE:-mean}   # mean | tau_logz (fixed_dataset.baseline)
LEVEL=${LEVEL:-seq_sum}
FORM=${FORM:-squared}
STD_ADV=${STD_ADV:-False}
HOLDOUT_PROMPTS=${HOLDOUT_PROMPTS:-256}     # fresh prompts (flag 1)
HOLDOUT_SAMPLE_PROMPTS=${HOLDOUT_SAMPLE_PROMPTS:-256}   # train prompts with a second fresh sample set (flag 2)
PROBE_PROMPTS=${PROBE_PROMPTS:-128}         # 128 x 16 = 2048 in-sample probe rows
STEPS=${STEPS:-200}
TEST_FREQ=${TEST_FREQ:-1}
VAL_N=${VAL_N:-4}
SAVE_FREQ=${SAVE_FREQ:-25}
HF_SAVE_FREQ=${HF_SAVE_FREQ:-5}
HF_DENSE=${HF_DENSE:-20}
TAG=${TAG:-}

PROMPT_DIR=${PROMPT_DIR:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/prompts_b${BSZ}_h${HOLDOUT_PROMPTS}_seed0}
BANK_DIR=${BANK_DIR:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed}
BANK=${BANK:-${BANK_DIR}/bank_b${BSZ}_n${N}_hp${HOLDOUT_PROMPTS}_hs${HOLDOUT_SAMPLE_PROMPTS}.pkl}
OUTPUT_DIR=${OUTPUT_DIR:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/runs}

# ---- master bank (README.md "Master bank"): MODE=master_build builds shards; SELECTION=<dir> trains on a subset
MASTER_DIR=${MASTER_DIR:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/master_dapo_n64}
MASTER_PROMPTS=${MASTER_PROMPTS:-${MASTER_DIR}/prompts/master_prompts.parquet}
CHUNK=${CHUNK:-512}            # prompts per shard
JOB_INDEX=${JOB_INDEX:-0}      # this build job takes shards c with c % NUM_JOBS == JOB_INDEX
NUM_JOBS=${NUM_JOBS:-1}
SELECTION=${SELECTION:-}       # dir with selection.json + monitor_{seen,fresh}.parquet (prepare_master_selection.py)
MASTER_SELECTION_ARG=()
if [ "${MODE}" = "master_build" ]; then
    N=${N_MASTER:-64}
    BSZ=${CHUNK}
    HOLDOUT_PROMPTS=0
    VAL_BEFORE=False
elif [ -n "${SELECTION}" ]; then
    read -r SEL_P SEL_K SEL_PROMPTS < <(python3 -c "import json; s = json.load(open('${SELECTION}/selection.json')); print(len(s['train']), s['k'], s['prompt_file'])")
    BSZ=${SEL_P}
    N=${SEL_K}
    HOLDOUT_PROMPTS=0          # held-out rows come from the selection, not from the dataloader
    PROMPT_DIR=${SELECTION}
    MASTER_PROMPTS=${SEL_PROMPTS}
    MASTER_SELECTION_ARG=(trainer.fixed_dataset.master_selection=${SELECTION}/selection.json)
fi

project_name=${PROJECT:-"kimi_k2_fixed"}
if [ "${MODE}" = "master_build" ]; then
    experiment_name="master_$(basename "${MASTER_DIR}")_job${JOB_INDEX}of${NUM_JOBS}"
    build_only=True
elif [ "${MODE}" = "build" ]; then
    experiment_name="bank_b${BSZ}_n${N}_hp${HOLDOUT_PROMPTS}_hs${HOLDOUT_SAMPLE_PROMPTS}"
    build_only=True
else
    experiment_name="k2fix_n${N}_bsz${BSZ}_tau${TAU}_${LEVEL}_${FORM}_fullbatch_lr${LR}_anchor${ANCHOR}"
    if [ "${BASELINE}" != "mean" ]; then experiment_name="${experiment_name}_bl${BASELINE}"; fi
    if [ -n "${SELECTION}" ]; then experiment_name="${experiment_name}_sel$(basename "${SELECTION}")"; fi
    build_only=False
fi
if [ -n "${TAG}" ]; then experiment_name="${experiment_name}_${TAG}"; fi

ENV_DIR=${ENV_DIR:-/n/holylabs/dam_lab/Lab/brachit/envs}
[ -r "${ENV_DIR}/bin/activate" ] || ENV_DIR=/n/netscratch/dam_lab/Everyone/brachit/share/envs
source "${ENV_DIR}/bin/activate"
VERL_ROOT=${VERL_ROOT:-/n/home08/brachit/coverage/verl}
PYEXTRAS=${PYEXTRAS:-$(dirname "${VERL_ROOT}")/pyextras}
[ -d "${PYEXTRAS}" ] || PYEXTRAS=/n/netscratch/dam_lab/Everyone/brachit/share/kimi_k2_hp128/pyextras
export PYTHONPATH=${VERL_ROOT}:${PYEXTRAS}:${PYTHONPATH:-}
cd "${VERL_ROOT}"

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

model_path=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct
train_data=${PROMPT_DIR}/bank_prompts.parquet
aime_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/aime1983_2024.parquet
val_files="[${aime_data},${PROMPT_DIR}/monitor_seen.parquet,${PROMPT_DIR}/monitor_fresh.parquet]"
if [ "${MODE}" = "master_build" ]; then
    train_data=${MASTER_PROMPTS}
    val_files="[${aime_data}]"   # not run (VAL_BEFORE=False, no training steps), but the trainer loads it
    MASTER_SELECTION_ARG=(trainer.fixed_dataset.master_dir=${MASTER_DIR} trainer.fixed_dataset.master_chunk_prompts=${CHUNK}
        trainer.fixed_dataset.master_job_index=${JOB_INDEX} trainer.fixed_dataset.master_num_jobs=${NUM_JOBS})
elif [ -n "${SELECTION}" ]; then
    train_data=${MASTER_PROMPTS}
fi
[ -f "${train_data}" ] || { echo "missing ${train_data}: run examples/kimi_k2/fixed_ref/prepare_prompts.py --out ${PROMPT_DIR}"; exit 1; }
if [ "${MODE}" = "train" ] && [ -z "${SELECTION}" ] && [ ! -f "${BANK}" ]; then echo "missing bank ${BANK}: run MODE=build first"; exit 1; fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=${STD_ADV} \
    algorithm.use_kl_in_reward=False \
    data.train_files=${train_data} \
    data.val_files="${val_files}" \
    data.shuffle=False \
    data.train_batch_size=${BSZ} \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.kimi_holdout_prompts=${HOLDOUT_PROMPTS} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-600} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BSZ} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.policy_loss.loss_mode=kimi_k2 \
    actor_rollout_ref.actor.policy_loss.kimi_tau=${TAU} \
    actor_rollout_ref.actor.policy_loss.kimi_target_level=${LEVEL} \
    actor_rollout_ref.actor.policy_loss.kimi_form=${FORM} \
    actor_rollout_ref.actor.policy_loss.kimi_logratio_clip=null \
    actor_rollout_ref.actor.policy_loss.kimi_probe_num_seqs=0 \
    actor_rollout_ref.actor.policy_loss.kimi_holdout_num_seqs=0 \
    actor_rollout_ref.actor.policy_loss.kimi_probe_per_step=False \
    actor_rollout_ref.actor.policy_loss.kimi_probe_micro_bsz=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.fixed_dataset.enable=True \
    trainer.fixed_dataset.path=${BANK} \
    trainer.fixed_dataset.build_only=${build_only} \
    trainer.fixed_dataset.anchor=${ANCHOR} \
    trainer.fixed_dataset.baseline=${BASELINE} \
    trainer.fixed_dataset.holdout_sample_prompts=${HOLDOUT_SAMPLE_PROMPTS} \
    trainer.fixed_dataset.probe_prompts=${PROBE_PROMPTS} \
    trainer.fixed_dataset.hf_save_freq=${HF_SAVE_FREQ} \
    trainer.fixed_dataset.hf_save_dense_until=${HF_DENSE} \
    trainer.val_before_train=${VAL_BEFORE:-True} \
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
    trainer.total_training_steps=${STEPS} \
    trainer.total_epochs=1 \
    "${MASTER_SELECTION_ARG[@]}"
