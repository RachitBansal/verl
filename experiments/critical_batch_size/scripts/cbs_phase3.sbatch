#!/bin/bash
#SBATCH --job-name=cbs-p3
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cbs-p3-%A_%a.out
#SBATCH --error=logs/cbs-p3-%A_%a.out
#SBATCH --array=0-11

# Phase 3: Multi-epoch / Algorithmic CBS
#
# Studies the "algorithmic critical batch" that emerges when ppo_epochs > 1:
# as the policy updates multiple times over the same stale rollout batch,
# the IS ratio drifts away from 1.0, clipping kicks in, and the effective
# learning signal per token consumed falls.
#
# Experimental design:
#   - Fix n_rollouts=8, vary (n_prompts, ppo_epochs)
#   - Use Phase 1 (ep=1) runs as the on-policy baseline
#   - Three iso-optimization "diagonals" where n_prompts * ppo_epochs = const:
#       256 units: (256,ep1)* vs (128,ep2) vs (64,ep4)
#       512 units: (512,ep1)* vs (256,ep2) vs (128,ep4) vs (64,ep8)
#      1024 units: (1024,ep1)* vs (512,ep2) vs (256,ep4) vs (128,ep8)
#   * ep=1 baselines are already available from Phase 1
#
# Key metrics to watch in wandb:
#   actor/pg_clipfrac  - rises as policy drifts across epochs; signals algorithmic limit
#   actor/ppo_kl       - KL between new and old policy per mini-batch
#   val/gsm8k/acc      - downstream accuracy used to compute S(B, ep) surface
#
# Array index → (n_prompts, ppo_epochs)
#  0: (128, 2)   iso-opt=256  with (256, ep1) from Phase 1
#  1: (128, 4)   iso-opt=512  with (512, ep1) from Phase 1
#  2: (128, 8)   iso-opt=1024 with (1024,ep1) from Phase 1
#  3: (256, 2)   iso-opt=512  with (512, ep1) from Phase 1
#  4: (256, 4)   iso-opt=1024 with (1024,ep1) from Phase 1
#  5: (256, 8)   deep multi-epoch at medium batch
#  6: (512, 2)   iso-opt=1024 with (1024,ep1) from Phase 1
#  7: (512, 4)   deep multi-epoch at large batch
#  8: (512, 8)   deep multi-epoch at large batch
#  9: (64,  4)   iso-opt=256  with (256, ep1) from Phase 1
# 10: (64,  8)   iso-opt=512  with (512, ep1) from Phase 1
# 11: (32,  8)   iso-opt=256 lower end

set -euo pipefail

N_ROLLOUTS=8

N_PROMPTS_ARRAY=(128 128 128 256 256 256 512 512 512  64  64  32)
PPO_EPOCHS_ARRAY=(  2   4   8   2   4   8   2   4   8   4   8   8)

IDX=$SLURM_ARRAY_TASK_ID
N_PROMPTS=${N_PROMPTS_ARRAY[$IDX]}
PPO_EPOCHS=${PPO_EPOCHS_ARRAY[$IDX]}

# mini-batch size: keep consistent with Phase 1 / 2 so gradient noise estimates
# are comparable; cap at 256 to avoid single-pass (K=1) which gives no B_noise signal.
if [ "${N_PROMPTS}" -le 256 ]; then
    MINI_BATCH_SIZE=${N_PROMPTS}
else
    MINI_BATCH_SIZE=256
fi

mkdir -p logs

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

echo "============================================================"
echo "CBS Phase 3 [${IDX}]: multi-epoch algorithmic CBS"
echo "  n_prompts=${N_PROMPTS}, n_rollouts=${N_ROLLOUTS}, ppo_epochs=${PPO_EPOCHS}"
echo "  iso_opt_units=$((N_PROMPTS * PPO_EPOCHS))"
echo "  Started at: $(date)"
echo "============================================================"

bash experiments/critical_batch_size/scripts/cbs_sweep.sh \
    --n_prompts   "${N_PROMPTS}" \
    --n_rollouts  "${N_ROLLOUTS}" \
    --ppo_epochs  "${PPO_EPOCHS}" \
    --mini_batch_size "${MINI_BATCH_SIZE}" \
    --phase       p3 \
    --total_epochs 5 \
    --test_freq   25 \
    --save_freq   100

echo "============================================================"
echo "CBS Phase 3 [${IDX}]: Completed at $(date)"
echo "============================================================"
