#!/usr/bin/env bash
#SBATCH --job-name=k2rot_smoke
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

# Smoke of sub-sampled training (fixed_dataset.rotate_k / max_update_rows) on 2 GPUs, on the master bank of a
# smoke_master_ext.sh run (24 prompts x 8 samples, selection 12 train x k=2, pooled N=32 baselines):
#   A: rotate_k=2, one update call per step            (2 steps)
#   B: rotate_k=2, max_update_rows=8 (3 calls/step)    (2 steps)  -> step 1 must match A (same rows, same grad)
#   C: rotate_k=8 = the whole pool (prompts with a held-out sample set: 6), max_update_rows=24, 2 steps
# Checks: grep "ROT_COMPARE" / SMOKE_OK in the log.
set -xeuo pipefail
if [ -n "${CANCEL_TWINS:-}" ]; then scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true; fi
ENV_DIR=${ENV_DIR:-/n/holylabs/dam_lab/Lab/brachit/envs}
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
SRC=${SRC:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/smoke_master_ext_57580539}
MASTER=${SRC}/master
SEL=${SRC}/sel
BASEL=${MASTER}/baselines/n32.npz
OUT=/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/smoke_rotate_${SLURM_JOB_ID}
mkdir -p ${OUT}
BSZ=12 NROLL=2
train=(
    algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False algorithm.use_kl_in_reward=False
    data.train_files=${MASTER}/prompts/master_prompts.parquet
    "data.val_files=[${SEL}/monitor_seen.parquet,${SEL}/monitor_fresh.parquet]"
    data.shuffle=False data.train_batch_size=${BSZ} data.max_prompt_length=512 data.max_response_length=2048
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
    trainer.fixed_dataset.enable=True trainer.fixed_dataset.build_only=False
    trainer.fixed_dataset.holdout_sample_prompts=2 trainer.fixed_dataset.probe_prompts=2
    trainer.fixed_dataset.hf_save_freq=0 trainer.fixed_dataset.hf_save_dense_until=0
    trainer.fixed_dataset.master_selection=${SEL}/selection.json
    trainer.fixed_dataset.baseline=mean trainer.fixed_dataset.baseline_file=${BASEL}
    trainer.critic_warmup=0 'trainer.logger=["console","wandb"]'
    trainer.project_name=kimi_k2_fixed_smoke trainer.n_gpus_per_node=2 trainer.nnodes=1
    trainer.save_freq=0 trainer.test_freq=0 trainer.total_epochs=1 trainer.total_training_steps=2
    trainer.val_before_train=False
)
run() {  # name rotate_k max_update_rows
python3 -m verl.trainer.main_ppo "${train[@]}" trainer.fixed_dataset.rotate_k=$2 \
    trainer.fixed_dataset.max_update_rows=$3 \
    trainer.experiment_name=smoke_rot_$1 trainer.default_local_dir=${OUT}/$1 2>&1 | tee ${OUT}/$1.log
}
run A 2 0
run B 2 8
run C 8 24

python - <<PYEOF
import re
def steps(path):
    out = {}
    for line in open(path):
        m = re.search(r"step:(\d+) - (.*)", line)
        if m and "actor/loss_exact" in line:
            d = {}
            for kv in m.group(2).split(" - "):
                k, _, v = kv.partition(":")
                try:
                    d[k.strip()] = float(v)
                except ValueError:
                    pass
            out[int(m.group(1))] = d
    return out
A, B, C = (steps("${OUT}/%s.log" % n) for n in "ABC")
keys = ["actor/loss_exact", "actor/grad_norm", "rotation/chunks", "rotation/rows", "rotation/epoch", "rotation/slot",
        "kimi_probe/reg_loss_before", "kimi_probe/reg_loss_after", "kimi_holdout_prompt/reg_loss_before",
        "kimi_holdout_prompt/reg_loss_after", "kimi_holdout_sample/reg_loss_after", "actor/lr"]
for s in (1, 2):
    for name, R in (("A", A), ("B", B), ("C", C)):
        print("ROT_COMPARE step", s, name, {k: R.get(s, {}).get(k) for k in keys})
a, b = A[1], B[1]
for k in ("actor/loss_exact", "actor/grad_norm", "kimi_holdout_prompt/reg_loss_after", "kimi_probe/reg_loss_after"):
    rel = abs(a[k] - b[k]) / max(abs(a[k]), 1e-12)
    print("ROT_COMPARE A vs B step 1", k, a[k], b[k], "rel diff", rel)
    assert rel < 2e-2, (k, a[k], b[k])
assert B[1]["rotation/chunks"] >= 3 and A[1]["rotation/chunks"] == 1 and C[1]["rotation/rows"] == 92
print("SMOKE_OK")
PYEOF
