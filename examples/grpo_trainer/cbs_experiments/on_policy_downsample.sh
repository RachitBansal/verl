#!/usr/bin/env bash
#SBATCH --job-name=cbs_downsample
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -xeuo pipefail

####################
# Downsample-rollouts CBS experiment.
# Generate N rollouts per prompt, compute GRPO advantages over ALL N (high-quality
# baseline), then downsample DSK of them per prompt for the actual gradient update.
# Decouples advantage-estimation quality (fixed at N) from gradient sample count (DSK).
# Usage: BSZ=128 N=64 DSK=8 LR=1e-5 sbatch on_policy_downsample.sh
####################

project_name="grpo_on_policy_cbs"
experiment_name="downsample_n${N}_dsk${DSK}_bsz${BSZ}_lr${LR}"

source /n/holylabs/dam_lab/Lab/brachit/envs/bin/activate
# verl is not pip-installed in this env; use the local source tree directly.
export PYTHONPATH=/n/home08/brachit/cbs-experiments/verl:${PYTHONPATH:-}

OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/brachit/cbs_k1_runs"

export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
# Project grpo_on_policy_cbs is owned by cmohri's entity; rachitbansal can't
# write to it. Route via the shared harvardml team (matches cbs_sweep.sh).
export WANDB_ENTITY=harvardml

# /tmp/ray on these compute nodes is mode 777 and accumulates other users'
# stale session_latest symlinks; ray.init() auto-attaches to those dead GCS
# addresses and times out. Use a job-private ray temp dir + stop any local
# ray we may have left behind.
export RAY_TMPDIR=/tmp/ray_brachit_${SLURM_JOB_ID}
mkdir -p "${RAY_TMPDIR}"
ray stop --force 2>/dev/null || true
sleep 5

n_resp_per_prompt=${N}
use_kl_loss=True
kl_loss_coeff=0.001
adv_estimator=grpo
use_kl_in_reward=False

train_prompt_bsz=${BSZ}
train_prompt_mini_bsz=$((train_prompt_bsz * n_resp_per_prompt))

gpu_memory_utilization=0.50
gen_tp=1
sp_size=1
max_prompt_length=1024
max_response_length=3072
data_truncation='left'

model_path=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct
train_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/dapo.parquet
test_data=/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/aime1983_2024.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=${train_data} \
    data.val_files=${test_data} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=${data_truncation} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coeff} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    +trainer.downsample_update_k=${DSK} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir="${OUTPUT_DIR}/${experiment_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_epochs=15
