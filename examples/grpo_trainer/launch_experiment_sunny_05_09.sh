#!/usr/bin/env bash
# Sunny's launcher (2026-05-09).
#
# Mirror of §2 of launch_experiments_clara_05_03.sh — parallel_avg combined
# SFT+RL on OLMo2-1B (50B pretrain) — but sweeping pretrain steps and using
# matched RL/SFT learning rates of 1e-6.
#
# Sweep: parallel_avg over pretrain steps 3k, 5k, 14k, 22k (OLMo2-1B-stage1-50B)
#   RL_LR=1e-6, SFT_LR=1e-6, SCALE_BATCH=False (_smbatch tag)
#   Exp names: OLMo2-1B_step{N}_parallel_avg_n32_rl1e-6_sft1e-6_smbatch
#   Job names: pavg-3k, pavg-5k, pavg-14k, pavg-22k
#   4 H100 x 72h per setting.
#
# Usage: bash examples/grpo_trainer/launch_experiment_sunny_05_09.sh

set -eo pipefail

# ============================================================================
# USER CONFIG
# ============================================================================
SLURM_ACCOUNT="kempner_barak_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
# ============================================================================

# Inner SLURM dispatcher
PARAVG_SLURM="examples/grpo_trainer/run_olmo2-1b_gsm_parallel_avg_slurm.sh"

step_to_suffix() {
    local step=$1
    if (( step >= 1000 )); then
        echo "$((step / 1000))k"
    else
        echo "${step}"
    fi
}

mkdir -p logs
COUNT=0

# ============================================================================
# parallel_avg, matched LR (RL=1e-6, SFT=1e-6) — 72h, 4 H100
#     Sweep over OLMo2-1B-stage1-50B pretrain steps: 3k, 5k, 14k, 22k.
#     SCALE_BATCH=False → _smbatch tag (same as Clara's §2).
# ============================================================================
PRETRAIN_STEPS=(3000 5000 14000 22000)

for STEP in "${PRETRAIN_STEPS[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting pavg-${SUFFIX} (parallel_avg, step ${STEP}, RL=1e-6 / SFT=1e-6) — 72h, 4 H100"
    sbatch --account=${SLURM_ACCOUNT} --partition=kempner_h100 \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="pavg-${SUFFIX}" --time=72:00:00 \
        --gpus-per-node=4 --cpus-per-task=96 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},RL_LR=1e-6,SFT_LR=1e-6,SCALE_BATCH=False,SAVE_FREQ=50,TEST_FREQ=25 \
        ${PARAVG_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
