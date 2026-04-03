#!/usr/bin/env bash
# Launch all SFT-only experiments for GSM8K and MATH (de-duplicated and duplicated)
# across 1B-50B and 1B-60B checkpoints.
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

# GSM8K data directories
GSM_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"
GSM_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k_duplicated"

# MATH data directories
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"
MATH_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_duplicated"

# SFT-only: all steps are SFT, no PPO
NUM_SFT_STEPS=50000
NUM_PPO_STEPS=0

# Checkpoint directories
CKPT_50B="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B"
CKPT_60B="/n/netscratch/barak_lab/Everyone/sqin/olmo/checkpoints/OLMo2-1B-stage1-60B"

# 1B-50B checkpoints
STEPS_50B=(1000 2000 3000 5000 6000 7000 10000 14000 22000)
# 1B-60B checkpoints (math only)
STEPS_60B=(5000 10000 15000 17000 22000 28000)

# Map step numbers to short suffixes for job names
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
# 1B-50B: GSM8K + MATH (de-duplicated and duplicated)
# ============================================================================
for STEP in "${STEPS_50B[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    # ---- GSM8K experiments ----

    echo "Submitting gsm-${SUFFIX} (GSM8K de-duplicated, 50B step ${STEP})"
    sbatch --account=${SLURM_ACCOUNT} --job-name="gsm-${SUFFIX}" \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${GSM_DEDUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_gsm" \
        ${GSM_SLURM}
    sleep 1

    echo "Submitting rgsm-${SUFFIX} (GSM8K duplicated, 50B step ${STEP})"
    sbatch --account=${SLURM_ACCOUNT} --job-name="rgsm-${SUFFIX}" \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${GSM_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rgsm" \
        ${GSM_SLURM}
    sleep 1

    # ---- MATH experiments ----

    echo "Submitting math-${SUFFIX} (MATH de-duplicated, 50B step ${STEP})"
    sbatch --account=${SLURM_ACCOUNT} --job-name="math-${SUFFIX}" \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_math" \
        ${MATH_SLURM}
    sleep 1

    echo "Submitting rmath-${SUFFIX} (MATH duplicated, 50B step ${STEP})"
    sbatch --account=${SLURM_ACCOUNT} --job-name="rmath-${SUFFIX}" \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_50B},SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
        ${MATH_SLURM}

    COUNT=$((COUNT + 4))
    sleep 1
done

# ============================================================================
# 1B-60B: MATH only (de-duplicated and duplicated)
# ============================================================================
for STEP in "${STEPS_60B[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting math60-${SUFFIX} (MATH de-duplicated, 60B step ${STEP})"
    sbatch --account=${SLURM_ACCOUNT} --job-name="math60-${SUFFIX}" \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_math" \
        ${MATH_SLURM}
    sleep 1

    echo "Submitting rmath60-${SUFFIX} (MATH duplicated, 60B step ${STEP})"
    sbatch --account=${SLURM_ACCOUNT} --job-name="rmath60-${SUFFIX}" \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
        ${MATH_SLURM}

    COUNT=$((COUNT + 2))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
