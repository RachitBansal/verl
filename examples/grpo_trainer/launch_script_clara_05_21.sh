#!/usr/bin/env bash
# Clara's launcher (2026-05-21): 4B SFT-only sweep, new pretrain steps.
#
# Mirrors §2 of launch_experiments_clara_04_29.sh but on pretrain steps 2k, 10k
# (instead of 5k, 14k). Same SFT-only setup otherwise.
#
# Sections:
#   1. SFT-only on 4B pretrain ckpts         (m4bs-*,   2 H100 x 48h)
#        SFT data: openmathinstruct2 (MATH dedupe, NOT rmath).
#        RL: none (NUM_PPO_STEPS=0).  lr=5e-4.  save_freq=500/test_freq=250.
#        MODEL_NAME=OLMo2-4B → exp name includes "4B":
#          OLMo2-4B_step{2000,10000}_interleave_twoloader_n32_sft_50000_ppo_0_math
#
# Usage: bash examples/grpo_trainer/launch_script_clara_05_21.sh

set -eo pipefail

# ============================================================================
# USER CONFIG — change these for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_barak_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
# ============================================================================

# Inner SLURM dispatcher
MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"

# Data directories
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

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
# §1 — SFT-only on 4B pretrain checkpoints (steps 2k, 10k) — 48h, 2 H100
#     50000 SFT steps, 0 PPO steps. lr=5e-4. save_freq=500 / test_freq=250.
#     MODEL_NAME=OLMo2-4B → exp name like
#       OLMo2-4B_step2000_interleave_twoloader_n32_sft_50000_ppo_0_math
#     Job names: m4bs-2k, m4bs-10k
# ============================================================================
STEPS_4B_SFT=(2000 10000)
SFTONLY_SFT=50000
SFTONLY_PPO=0

for STEP in "${STEPS_4B_SFT[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting m4bs-${SUFFIX} (4B SFT-only on MATH dedupe, step ${STEP}) — 48h, 2 H100"
    sbatch --account=${SLURM_ACCOUNT} --partition=kempner_h100 \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="m4bs-${SUFFIX}" --time=48:00:00 \
        --gpus-per-node=2 --cpus-per-task=48 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_4B},MODEL_NAME=OLMo2-4B,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${SFTONLY_SFT},NUM_PPO_STEPS=${SFTONLY_PPO},SAVE_FREQ=500,TEST_FREQ=250,RL_LR=5e-4,EXP_SUFFIX="interleave_twoloader_n32_sft_${SFTONLY_SFT}_ppo_${SFTONLY_PPO}_math" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
