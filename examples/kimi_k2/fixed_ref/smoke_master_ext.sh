#!/usr/bin/env bash
#SBATCH --job-name=k2ext_smoke
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# End-to-end smoke of large-N baselines on 2 GPUs: a base master bank (24 prompts x 8, seed 0, with reference
# log-probs), a baseline-only extension (x 24 samples, offset 8, rollout seed 1, no reference pass) and, as a
# control, an extension that reuses seed 0 (x 8, offset 32); compute_master_baselines.py pools base + extension
# (N = 32, checks cross-bank duplicates); then 3 steps with BASELINE=tau_logz and 2 with mean from the npz.
# Checks (in the log): "baseline-only bank", "re-baselined", bank_baseline/n_samples 32, SMOKE_OK.
set -xeuo pipefail
# twin submissions under one --job-name: the first to start cancels the pending twins
if [ -n "${CANCEL_TWINS:-}" ]; then scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true; fi
ENV_DIR=${ENV_DIR:-/n/holylabs/dam_lab/Lab/brachit/envs}
[ -r "${ENV_DIR}/bin/activate" ] || ENV_DIR=/n/netscratch/dam_lab/Everyone/brachit/share/envs
source "${ENV_DIR}/bin/activate"
export PYTHONPATH=/n/home08/brachit/coverage/verl:/n/home08/brachit/coverage/pyextras:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_${SLURM_JOB_ID}; export HF_DATASETS_CACHE=/tmp/hf_${SLURM_JOB_ID}/datasets; mkdir -p "${HF_DATASETS_CACHE}"
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}; mkdir -p "${WANDB_DIR}"; export WANDB_MODE=offline
export RAY_TMPDIR=/tmp/ray_brachit_${SLURM_JOB_ID}; mkdir -p "${RAY_TMPDIR}"
ray stop --force 2>/dev/null || true; sleep 3
cd /n/home08/brachit/coverage/verl

model_path=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct
OUT=/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/smoke_master_ext_${SLURM_JOB_ID}
MASTER=${OUT}/master
SEL=${OUT}/sel
SMOKE_PROMPTS=${SMOKE_PROMPTS:-24}   # master prompts (6 shards of 4)
SMOKE_RESP=${SMOKE_RESP:-2048}       # long enough that some groups have mixed rewards (non-zero advantages)
python examples/kimi_k2/fixed_ref/prepare_master_prompts.py --out ${MASTER}/prompts --limit ${SMOKE_PROMPTS} --max-prompt-length 512

EXT=${OUT}/ext_n24_s1
EXT0=${OUT}/ext_n8_s0_control
BASEL=${MASTER}/baselines/n32.npz

# 1) builds (one job each, shards of 4 prompts): base (8 samples, seed 0), extension (24, seed 1, no ref pass),
#    control extension (8, seed 0 = the base bank's seed)
BSZ=4 VALB=False
mkdir -p ${SEL}
build_bank() {  # dir n offset seed ref
NROLL=$2
build=(
    algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False algorithm.use_kl_in_reward=False
    data.train_files=${MASTER}/prompts/master_prompts.parquet
    data.shuffle=False data.train_batch_size=${BSZ} data.max_prompt_length=512 data.max_response_length=${SMOKE_RESP}
    data.filter_overlong_prompts=True data.truncation=left data.kimi_holdout_prompts=0
    actor_rollout_ref.model.path=${model_path} actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${BSZ} actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.policy_loss.loss_mode=kimi_k2 actor_rollout_ref.actor.policy_loss.kimi_tau=0.03
    actor_rollout_ref.actor.policy_loss.kimi_probe_num_seqs=0 actor_rollout_ref.actor.policy_loss.kimi_holdout_num_seqs=0
    actor_rollout_ref.actor.policy_loss.kimi_probe_per_step=False
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=${NROLL}
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    trainer.fixed_dataset.enable=True
        trainer.fixed_dataset.holdout_sample_prompts=2 trainer.fixed_dataset.probe_prompts=2
    trainer.fixed_dataset.hf_save_freq=0 trainer.fixed_dataset.hf_save_dense_until=2
    trainer.critic_warmup=0 'trainer.logger=["console","wandb"]'
    trainer.project_name=kimi_k2_fixed_smoke trainer.n_gpus_per_node=2 trainer.nnodes=1
    trainer.save_freq=3 trainer.test_freq=1 trainer.total_training_steps=3 trainer.total_epochs=1
    trainer.val_before_train=${VALB}
)
python3 -m verl.trainer.main_ppo "${build[@]}" trainer.fixed_dataset.build_only=True \
    "data.val_files=[${MASTER}/prompts/master_prompts.parquet]" \
    trainer.fixed_dataset.master_dir=$1 trainer.fixed_dataset.master_chunk_prompts=4 \
    trainer.fixed_dataset.master_job_index=0 trainer.fixed_dataset.master_num_jobs=1 \
    trainer.fixed_dataset.master_sample_offset=$3 actor_rollout_ref.rollout.seed=$4 \
    trainer.fixed_dataset.master_ref_log_probs=$5 \
    trainer.experiment_name=smoke_ext_build_$(basename $1) trainer.default_local_dir=${OUT}/build_$(basename $1)
}
build_bank ${MASTER} 8 0 0 True
build_bank ${EXT} 24 8 1 False
build_bank ${EXT0} 8 32 0 False
cat ${MASTER}/summary.json ${EXT}/summary.json; cat ${EXT}/build_config.json ${EXT}/manifest.json | head -30

# 2) pooled baselines (N = 32) + duplicate checks: extension vs base must have none; report the seed-0 control
python examples/kimi_k2/fixed_ref/compute_master_baselines.py --master ${MASTER} --master ${EXT} --out ${BASEL} \
    --check-overlap ${SMOKE_PROMPTS}
python - <<PYEOF
import sys; sys.path.insert(0, "examples/kimi_k2/fixed_ref")
from compute_master_baselines import overlap
print("SEED-0 CONTROL overlap (base vs extension with the same seed):", overlap(["${MASTER}", "${EXT0}"], list(range(${SMOKE_PROMPTS}))))
print("SEED-1 overlap (base vs extension):", overlap(["${MASTER}", "${EXT}"], list(range(${SMOKE_PROMPTS}))))
PYEOF
python - <<PYEOF
from verl.trainer.ppo import kimi_master_bank as mb
try:
    mb.gather_rows("${EXT}", [(0, [0])]); raise SystemExit("extension must not load for training")
except AssertionError as e:
    print("baseline-only bank refused for training:", e)
PYEOF

# 3) selection on the base bank: 12 train prompts x k=2 (samples 0,1), 4 fresh prompts, 2 fresh-sample prompts
python examples/kimi_k2/fixed_ref/prepare_master_selection.py --master ${MASTER} --out ${SEL} --train 12 --k 2 \
    --holdout-prompts 4 --holdout-sample-prompts 2 --monitor 2

# 4) train with the pooled baselines: tau_logz (3 steps), then mean (2 steps, fresh run dir)
BSZ=12 NROLL=2 VALB=True
train=(
    algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False algorithm.use_kl_in_reward=False
    data.train_files=${MASTER}/prompts/master_prompts.parquet
    "data.val_files=[${SEL}/monitor_seen.parquet,${SEL}/monitor_fresh.parquet]"
    data.shuffle=False data.train_batch_size=${BSZ} data.max_prompt_length=512 data.max_response_length=${SMOKE_RESP}
    data.filter_overlong_prompts=True data.truncation=left data.kimi_holdout_prompts=0
    actor_rollout_ref.model.path=${model_path} actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${BSZ} actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.policy_loss.loss_mode=kimi_k2 actor_rollout_ref.actor.policy_loss.kimi_tau=0.03
    actor_rollout_ref.actor.policy_loss.kimi_probe_num_seqs=0 actor_rollout_ref.actor.policy_loss.kimi_holdout_num_seqs=0
    actor_rollout_ref.actor.policy_loss.kimi_probe_per_step=False
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=${NROLL}
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    trainer.fixed_dataset.enable=True
        trainer.fixed_dataset.holdout_sample_prompts=2 trainer.fixed_dataset.probe_prompts=2
    trainer.fixed_dataset.hf_save_freq=0 trainer.fixed_dataset.hf_save_dense_until=2
    trainer.critic_warmup=0 'trainer.logger=["console","wandb"]'
    trainer.project_name=kimi_k2_fixed_smoke trainer.n_gpus_per_node=2 trainer.nnodes=1
    trainer.save_freq=3 trainer.test_freq=1 trainer.total_epochs=1
    trainer.val_before_train=${VALB}
)
for BL in tau_logz mean; do
STEPS_BL=$([ "${BL}" = tau_logz ] && echo 3 || echo 2)
python3 -m verl.trainer.main_ppo "${train[@]}" trainer.fixed_dataset.build_only=False \
    trainer.fixed_dataset.master_selection=${SEL}/selection.json \
    trainer.fixed_dataset.baseline=${BL} trainer.fixed_dataset.baseline_file=${BASEL} \
    trainer.total_training_steps=${STEPS_BL} \
    trainer.experiment_name=smoke_ext_train_${BL} trainer.default_local_dir=${OUT}/train_${BL}
done
echo SMOKE_OK
