#!/usr/bin/env bash
# Rachit's experiment launcher.
#
# Sections:
#   1. 60B base + RMATH SFT-only (resume)    (rm60-*,    2 GPU x 24h)
#        SFT data: openmathinstruct2_duplicated (RMATH)   RL: none
#   2. MATH RL on 60BMATH pretrain ckpts     (m60rl-*,   4 GPU x 72h)
#        SFT: none (NUM_SFT_STEPS=0)   RL data: openmathinstruct2 (MATH dedupe)
#
# Comment out steps in the STEPS_*=( ... ) arrays to split work.
#
# Usage: bash examples/grpo_trainer/launch_gsm_sft_experiments_rachit.sh

set -eo pipefail

# ============================================================================
# USER CONFIG — change these for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_dam_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
# ============================================================================

# Inner SLURM dispatchers
MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"

# Data directories
MATH_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_duplicated"

# Checkpoint directories
CKPT_60B="/n/netscratch/barak_lab/Everyone/sqin/olmo/checkpoints/OLMo2-1B-stage1-60B"

# SFT-only defaults (used in section 1)
NUM_SFT_STEPS=50000
NUM_PPO_STEPS=0

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
# §1 — 60B base + RMATH SFT (5k, 10k) — RESUME for another 24h, 2 GPU
#     SFT data: openmathinstruct2_duplicated (RMATH)
#     RL: none (NUM_PPO_STEPS=0, SFT-only run).
#     save_freq=500 / test_freq=250 (long SFT runs).
#     Job names: rm60-5k, rm60-10k
# ============================================================================
STEPS_60B_RMATH=(5000 10000)

for STEP in "${STEPS_60B_RMATH[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting rm60-${SUFFIX} (60B RMATH SFT, step ${STEP}) — resume, 24h"
    sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="rm60-${SUFFIX}" --time=24:00:00 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},SAVE_FREQ=500,TEST_FREQ=250,EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

# ============================================================================
# §2 — MATH RL on 60BMATH pretrain checkpoints — 72h, 4 GPU
#     SFT: none (NUM_SFT_STEPS=0; SFT_DATA_DIR set for completeness but unused).
#     RL data: openmathinstruct2 (MATH dedupe, hardcoded in math interleave inner script).
#     save_freq=50 / test_freq=25.
#     Job names: m60rl-5k, m60rl-10k, m60rl-14k
# ============================================================================
STEPS_60BMATH_RL=(5000 10000 14000)
INTERLEAVE_SFT=0
INTERLEAVE_PPO=50000

for STEP in "${STEPS_60BMATH_RL[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting m60rl-${SUFFIX} (60BMATH MATH RL, step ${STEP}) — 72h, 4 GPU"
    sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="m60rl-${SUFFIX}" --time=72:00:00 \
        --gpus-per-node=4 --cpus-per-task=96 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},SAVE_FREQ=50,TEST_FREQ=25,EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
