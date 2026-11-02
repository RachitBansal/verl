#!/usr/bin/env bash
#SBATCH --job-name=k2master_smoke
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# End-to-end smoke of the master bank on 2 GPUs: build a 24-prompt x 8-sample master bank as 6 shards split over
# two sequential build "jobs" (striding + resume + manifest), choose a selection (12 train prompts x k=2, 4 fresh
# prompts, 2 fresh-sample prompts), then 3 full-batch steps loaded from it.  Checks (in the log): "master bank
# complete", "loaded ... rows for selection", kimi_*/delta_rms_before ~ 0 at step 1 (anchor = stored log-probs).
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
OUT=/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/smoke_master_${SLURM_JOB_ID}
MASTER=${OUT}/master
SEL=${OUT}/sel
SMOKE_PROMPTS=${SMOKE_PROMPTS:-24}   # master prompts (6 shards of 4)
SMOKE_RESP=${SMOKE_RESP:-2048}       # long enough that some groups have mixed rewards (non-zero advantages)
python examples/kimi_k2/fixed_ref/prepare_master_prompts.py --out ${MASTER}/prompts --limit ${SMOKE_PROMPTS} --max-prompt-length 512

# 1) build: SMOKE_PROMPTS prompts x 8 samples in shards of 4 prompts, split over two "jobs" run one after the other
mkdir -p ${SEL}; for J in 0 1; do
BSZ=4 NROLL=8 VALB=False
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
    trainer.fixed_dataset.baseline=${BASELINE:-mean}
    trainer.fixed_dataset.holdout_sample_prompts=2 trainer.fixed_dataset.probe_prompts=2
    trainer.fixed_dataset.hf_save_freq=0 trainer.fixed_dataset.hf_save_dense_until=2
    trainer.critic_warmup=0 'trainer.logger=["console","wandb"]'
    trainer.project_name=kimi_k2_fixed_smoke trainer.n_gpus_per_node=2 trainer.nnodes=1
    trainer.save_freq=3 trainer.test_freq=1 trainer.total_training_steps=3 trainer.total_epochs=1
    trainer.val_before_train=${VALB}
)
python3 -m verl.trainer.main_ppo "${build[@]}" trainer.fixed_dataset.build_only=True \
    "data.val_files=[${MASTER}/prompts/master_prompts.parquet]" \
    trainer.fixed_dataset.master_dir=${MASTER} trainer.fixed_dataset.master_chunk_prompts=4 \
    trainer.fixed_dataset.master_job_index=${J} trainer.fixed_dataset.master_num_jobs=2 \
    trainer.experiment_name=smoke_master_build_${J} trainer.default_local_dir=${OUT}/build_${J}
done
ls -la ${MASTER} ${MASTER}/shards; cat ${MASTER}/summary.json

# 2) selection: 12 train prompts x k=2 (samples 0,1), 4 fresh prompts, 2 fresh-sample prompts (samples 2,3)
python examples/kimi_k2/fixed_ref/prepare_master_selection.py --master ${MASTER} --out ${SEL} --train 12 --k 2 \
    --holdout-prompts 4 --holdout-sample-prompts 2 --monitor 2

# 3) train 3 steps on the selection
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
    trainer.fixed_dataset.baseline=${BASELINE:-mean}
    trainer.fixed_dataset.holdout_sample_prompts=2 trainer.fixed_dataset.probe_prompts=2
    trainer.fixed_dataset.hf_save_freq=0 trainer.fixed_dataset.hf_save_dense_until=2
    trainer.critic_warmup=0 'trainer.logger=["console","wandb"]'
    trainer.project_name=kimi_k2_fixed_smoke trainer.n_gpus_per_node=2 trainer.nnodes=1
    trainer.save_freq=3 trainer.test_freq=1 trainer.total_training_steps=3 trainer.total_epochs=1
    trainer.val_before_train=${VALB}
)
python3 -m verl.trainer.main_ppo "${train[@]}" trainer.fixed_dataset.build_only=False \
    trainer.fixed_dataset.master_selection=${SEL}/selection.json \
    trainer.experiment_name=smoke_master_train trainer.default_local_dir=${OUT}/train
ls ${OUT}/train
echo SMOKE_OK
