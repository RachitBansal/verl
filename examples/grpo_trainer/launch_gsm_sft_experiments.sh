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
# ============================================================================

GSM_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_gsm_interleave_slurm.sh"
MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"
COMBINED_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_combined_slurm.sh"

# Data directories
GSM_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
GSM_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k_duplicated"
MATH_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_duplicated"
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

# SFT-only: all steps are SFT, no PPO
NUM_SFT_STEPS=50000
NUM_PPO_STEPS=0

# Checkpoint directories
CKPT_50B="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B"
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
# 1. Combined SFT+RL on GSM (1B-50B, step 10k) — 1:1 loss weight — 48 hours
#    SFT data: RGSM (duplicated), RL data: GSM (de-duplicated)
# ============================================================================
STEP=10000
SUFFIX=$(step_to_suffix $STEP)

echo "Submitting combined-${SUFFIX} (combined SFT+RL, 50B step ${STEP}) — 48h"
sbatch --account=${SLURM_ACCOUNT} --job-name="comb-${SUFFIX}" --time=48:00:00 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},RL_LOSS_WEIGHT=1.0,SFT_LOSS_WEIGHT=1.0,LR_SCALE=1.0 \
    ${COMBINED_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# 2. Interleaved SFT+RL on GSM (1B-50B, step 7k) — 20:1 ratio — 36 hours
#    SFT data: RGSM (duplicated), RL data: GSM (de-duplicated)
# ============================================================================
STEP=7000
SUFFIX=$(step_to_suffix $STEP)
INTERLEAVE_SFT=20
INTERLEAVE_PPO=1

echo "Submitting rgsm-interleave-${SUFFIX} (RGSM SFT + GSM RL, 50B step ${STEP}) — 36h"
sbatch --account=${SLURM_ACCOUNT} --job-name="rgsm-il-${SUFFIX}" --time=36:00:00 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${GSM_DUP},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_rgsm" \
    ${GSM_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# 3. Resume 14k RGSM (1B-50B) — 12 hours
# ============================================================================
STEP=14000
SUFFIX=$(step_to_suffix $STEP)

echo "Submitting rgsm-${SUFFIX} (GSM8K duplicated, 50B step ${STEP}) — resume, 12h"
sbatch --account=${SLURM_ACCOUNT} --job-name="rgsm-${SUFFIX}" --time=12:00:00 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${GSM_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rgsm" \
    ${GSM_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# 4. Resume all RMATH (1B-50B) — 2 days
# ============================================================================
STEPS_RMATH_50B=(1000 2000 3000 5000 6000 7000 10000 14000 22000)

for STEP in "${STEPS_RMATH_50B[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting rmath-${SUFFIX} (MATH duplicated, 50B step ${STEP}) — resume, 48h"
    sbatch --account=${SLURM_ACCOUNT} --job-name="rmath-${SUFFIX}" --time=48:00:00 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

# ============================================================================
# 5. Resume 60B MATH (de-duplicated) — all except 17k — 36 hours
# ============================================================================
STEPS_60B_MATH=(5000 10000 15000 22000 28000)

for STEP in "${STEPS_60B_MATH[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting math60-${SUFFIX} (MATH de-duplicated, 60B step ${STEP}) — resume, 36h"
    sbatch --account=${SLURM_ACCOUNT} --job-name="math60-${SUFFIX}" --time=36:00:00 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_math" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

# ============================================================================
# 6. Resume 60B RMATH (duplicated) — 5k, 10k, 15k — 48 hours
# ============================================================================
STEPS_60B_RMATH=(5000 10000 15000)

for STEP in "${STEPS_60B_RMATH[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting rmath60-${SUFFIX} (MATH duplicated, 60B step ${STEP}) — resume, 48h"
    sbatch --account=${SLURM_ACCOUNT} --job-name="rmath60-${SUFFIX}" --time=48:00:00 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
