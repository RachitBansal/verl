#!/usr/bin/env bash
#SBATCH --job-name=k2fix_smoke
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:50:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# End-to-end smoke of trainer.fixed_dataset on 2 GPUs: build a tiny bank (8 train + 2 held-out prompts x 4,
# 2 fresh-sample prompts, 2 probe prompts), then 3 full-batch steps on it with validation on the monitor files,
# a full checkpoint and HF snapshots.  Checks (in the log): old_log_probs frozen (kimi_*/delta_rms_before at step 1
# ~ 0 and the step-2 *_before equals the step-1 *_after), both held-out prefixes present, bank summary logged.
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
OUT=/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/smoke_${SLURM_JOB_ID}
PROMPT_DIR=${OUT}/prompts
python examples/kimi_k2/fixed_ref/prepare_prompts.py --out ${PROMPT_DIR} --bank 8 --holdout 2 --monitor 4 --max-prompt-length 512
BANK=${OUT}/bank.pkl

common=(
    algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False algorithm.use_kl_in_reward=False
    data.train_files=${PROMPT_DIR}/bank_prompts.parquet
    "data.val_files=[${PROMPT_DIR}/monitor_seen.parquet,${PROMPT_DIR}/monitor_fresh.parquet]"
    data.shuffle=False data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=512
    data.filter_overlong_prompts=True data.truncation=left data.kimi_holdout_prompts=2
    actor_rollout_ref.model.path=${model_path} actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.policy_loss.loss_mode=kimi_k2 actor_rollout_ref.actor.policy_loss.kimi_tau=0.03
    actor_rollout_ref.actor.policy_loss.kimi_probe_num_seqs=0 actor_rollout_ref.actor.policy_loss.kimi_holdout_num_seqs=0
    actor_rollout_ref.actor.policy_loss.kimi_probe_per_step=False
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    trainer.fixed_dataset.enable=True trainer.fixed_dataset.path=${BANK}
    trainer.fixed_dataset.baseline=${BASELINE:-mean}
    trainer.fixed_dataset.holdout_sample_prompts=2 trainer.fixed_dataset.probe_prompts=2
    trainer.fixed_dataset.hf_save_freq=0 trainer.fixed_dataset.hf_save_dense_until=2
    trainer.critic_warmup=0 'trainer.logger=["console","wandb"]'
    trainer.project_name=kimi_k2_fixed_smoke trainer.n_gpus_per_node=2 trainer.nnodes=1
    trainer.save_freq=3 trainer.test_freq=1 trainer.total_training_steps=3 trainer.total_epochs=1
    trainer.val_before_train=True
)
# 1) build only
python3 -m verl.trainer.main_ppo "${common[@]}" trainer.fixed_dataset.build_only=True \
    trainer.experiment_name=smoke_build trainer.default_local_dir=${OUT}/build
ls -la ${BANK} ${BANK}.summary.json; cat ${BANK}.summary.json | head -40
# 2) train 3 steps on the saved bank (separate process: exercises the load path)
python3 -m verl.trainer.main_ppo "${common[@]}" trainer.fixed_dataset.build_only=False \
    trainer.experiment_name=smoke_train trainer.default_local_dir=${OUT}/train
ls ${OUT}/train
echo SMOKE_OK
