#!/usr/bin/env bash
# Rachit's experiment launcher.
#
# Sections:
#   1. MATH RL on 4B pretrain ckpts          (4brl-*,    4 GPU x 72h)
#        SFT: none (NUM_SFT_STEPS=0)   RL data: openmathinstruct2 (MATH dedupe)
#
# Comment out steps in STEPS_*=( ... ) arrays to split work between collaborators.
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
CKPT_4B="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-4B-stage1-50B"

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
# §1 — MATH RL on 4B pretrain checkpoints — 72h, 4 GPU
#     Pure RL (no SFT) on the 4B-stage1-50B model.
#     SFT: none (NUM_SFT_STEPS=0; SFT_DATA_DIR set for completeness but unused).
#     RL data: openmathinstruct2 (MATH dedupe, hardcoded in math interleave inner script).
#     save_freq=50 / test_freq=25.
#     MODEL_NAME=OLMo2-4B → exp name like OLMo2-4B_step5000_interleave_twoloader_n32_sft_0_ppo_50000_rmath
#     Job names: 4brl-5k, 4brl-14k
# ============================================================================
STEPS_4B_RL=(5000 14000)
INTERLEAVE_SFT=0
INTERLEAVE_PPO=50000

for STEP in "${STEPS_4B_RL[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting 4brl-${SUFFIX} (4B MATH RL, step ${STEP}) — 72h, 4 GPU"
    sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="4brl-${SUFFIX}" --time=72:00:00 \
        --gpus-per-node=4 --cpus-per-task=96 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_4B},MODEL_NAME=OLMo2-4B,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},SAVE_FREQ=50,TEST_FREQ=25,EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
