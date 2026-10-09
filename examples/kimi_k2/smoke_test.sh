#!/usr/bin/env bash
#SBATCH --job-name=k2_smoke
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# End-to-end smoke test of loss_mode=kimi_k2 + solve tracking on 2 GPUs (DP=2 exercises the FSDP
# lock-step requirement of the probe forwards). 3 tiny steps, no validation, wandb offline.
set -xeuo pipefail
source /n/holylabs/dam_lab/Lab/brachit/envs/bin/activate
export PYTHONPATH=/n/home08/brachit/coverage/verl:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_${SLURM_JOB_ID}; export HF_DATASETS_CACHE=/tmp/hf_${SLURM_JOB_ID}/datasets; mkdir -p "${HF_DATASETS_CACHE}"
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}; mkdir -p "${WANDB_DIR}"; export WANDB_MODE=offline
export RAY_TMPDIR=/tmp/ray_brachit_${SLURM_JOB_ID}; mkdir -p "${RAY_TMPDIR}"
ray stop --force 2>/dev/null || true; sleep 3

model_path=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct
train_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/dapo.parquet
test_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/aime1983_2024.parquet
OUT=/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_runs/smoke_${SLURM_JOB_ID}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_files=${train_data} data.val_files=${test_data} \
    data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=512 \
    data.filter_overlong_prompts=True data.truncation=left \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.policy_loss.loss_mode=kimi_k2 \
    actor_rollout_ref.actor.policy_loss.kimi_tau=0.01 \
    actor_rollout_ref.actor.policy_loss.kimi_probe_num_seqs=16 \
    actor_rollout_ref.actor.policy_loss.kimi_holdout_num_seqs=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=kimi_k2_smoke trainer.experiment_name=smoke_${SLURM_JOB_ID} \
    trainer.default_local_dir=${OUT} \
    trainer.n_gpus_per_node=2 trainer.nnodes=1 \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.total_training_steps=3 trainer.total_epochs=1
