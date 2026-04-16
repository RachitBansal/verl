#!/usr/bin/env bash
# Resume SFT-only experiments and launch new interleaved/combined runs.
#
# Usage: bash examples/grpo_trainer/launch_gsm_sft_experiments.sh

set -euo pipefail

# ============================================================================
# USER CONFIG — change these 3 lines for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_barak_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"  # email for SLURM failure notifications
# ============================================================================

MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"

# Data directories
MATH_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_duplicated"
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

# SFT-only: all steps are SFT, no PPO
NUM_SFT_STEPS=50000
NUM_PPO_STEPS=0

# Checkpoint directories
CKPT_60B="/n/netscratch/barak_lab/Everyone/sqin/olmo/checkpoints/OLMo2-1B-stage1-60B"

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
# Resume 60B RMATH (duplicated) — 5k, 10k, 15k — 48 hours
# ============================================================================
STEPS_60B_RMATH=(5000 10000 15000)

for STEP in "${STEPS_60B_RMATH[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting rmath60-${SUFFIX} (MATH duplicated, 60B step ${STEP}) — resume, 48h"
    sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} --job-name="rmath60-${SUFFIX}" --time=48:00:00 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

# ============================================================================
# Resume 60B MATH (de-duplicated) — step 15k only — 48 hours
# ============================================================================
STEP=15000
SUFFIX=$(step_to_suffix $STEP)

echo "Submitting math60-${SUFFIX} (MATH de-duplicated, 60B step ${STEP}) — resume, 48h"
sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} --job-name="math60-${SUFFIX}" --time=48:00:00 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_math" \
    ${MATH_SLURM}
COUNT=$((COUNT + 1))
sleep 1

echo ""
echo "Submitted ${COUNT} jobs total."
