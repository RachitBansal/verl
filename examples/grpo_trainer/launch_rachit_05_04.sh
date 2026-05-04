#!/usr/bin/env bash
# Rachit's launcher (2026-05-05).
#
# RL on OLMo2-1B-60BMATH pretrain checkpoints — 56h, 4 H100 each.
#   Inner script:  examples/grpo_trainer/run_olmo2-1b_openmath.sh
#                  (auto-derives OLMO_CHECKPOINT from STEP_NUM:
#                   /n/netscratch/barak_lab/Everyone/sqin/olmo/checkpoints/
#                     OLMo2-1B-stage1-60B/step${STEP_NUM}-hf)
#   SLURM wrapper: examples/grpo_trainer/run_olmo2-1b_openmath_slurm.sh
#                  (forwards STEP_NUM as $1 → exported into the inner script)
#   Pretrain steps: 5k, 10k, 14k
#   save_freq=50, test_freq=25 (configured in inner script)
#   Exp name: olmo2_1b_60bmath_step{N}_omi_n32
#   Job names: rl-5k, rl-10k, rl-14k
#
# Usage: bash examples/grpo_trainer/launch_rachit_05_04.sh
#        (runnable from anywhere — the script cd's into VERL_DIR itself)

set -eo pipefail

# ============================================================================
# USER CONFIG — change these for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_barak_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
# ============================================================================

cd "${VERL_DIR}"
SLURM_WRAPPER="examples/grpo_trainer/run_olmo2-1b_openmath_slurm.sh"

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

PRETRAIN_STEPS=(5000 10000 14000)

for STEP in "${PRETRAIN_STEPS[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting rl-${SUFFIX} (RL on OLMo2-1B-60BMATH step ${STEP}) — 56h, 4 H100"
    sbatch --account=${SLURM_ACCOUNT} --partition=kempner_h100 \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="rl-${SUFFIX}" --time=56:00:00 \
        --gpus-per-node=4 --cpus-per-task=96 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP} \
        ${SLURM_WRAPPER} ${STEP}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
