#!/usr/bin/env bash
# Multi-section experiment launcher.
#
# Sections:
#   1. 60B base + RMATH SFT-only             (rm60-*,    2 GPU x 48h)
#   2. GSM RL on already-SFTed 1B ckpts      (srl-*,     2 GPU x 24h)
#   3. Curriculum SFT->RL on 10k pretrain    (curd/curn, 2 GPU x 24h)
#   4. MATH RL on 60BMATH pretrain ckpts     (m60rl-*,   4 GPU x 72h)
#
# Splitting work between collaborators: each section has a STEPS=( ... )
# array near the top. Comment out the steps you do NOT want to run, then
# `bash examples/grpo_trainer/launch_gsm_sft_experiments.sh`.
#
# Usage: bash examples/grpo_trainer/launch_gsm_sft_experiments.sh

set -eo pipefail

# ============================================================================
# USER CONFIG — change these for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_barak_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
# ============================================================================

# Inner SLURM dispatchers
MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"
GSM_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_gsm_interleave_slurm.sh"
CURRICULUM_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_curriculum_slurm.sh"

# Data directories
MATH_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_duplicated"

# Checkpoint directories
CKPT_60B="/n/netscratch/barak_lab/Everyone/sqin/olmo/checkpoints/OLMo2-1B-stage1-60B"
SFTED_GSM_BASE="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

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
# §1 — 60B base + RMATH SFT (5k, 10k) — 48h, 2 GPU
#     SFT data: openmathinstruct2_duplicated (RMATH)
#     Job names: rm60-5k, rm60-10k
#
#     ALREADY RUNNING — left in place for reference; uncomment to resubmit.
# ============================================================================
# STEPS_60B_RMATH=(5000 10000)
#
# for STEP in "${STEPS_60B_RMATH[@]}"; do
#     SUFFIX=$(step_to_suffix $STEP)
#
#     echo "Submitting rm60-${SUFFIX} (60B RMATH SFT, step ${STEP}) — 48h"
#     sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
#         --job-name="rm60-${SUFFIX}" --time=48:00:00 \
#         --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},EXP_SUFFIX="interleave_twoloader_n32_sft_${NUM_SFT_STEPS}_ppo_${NUM_PPO_STEPS}_rmath" \
#         ${MATH_SLURM}
#     COUNT=$((COUNT + 1))
#     sleep 1
# done

# ============================================================================
# §2 — GSM RL on already-SFTed 1B checkpoints — 24h, 2 GPU
#     Resume from hf_model/step2000 of the SFT-only GSM runs and run pure RL.
#     STEP_NUM is used as a label only; OLMO_CHECKPOINT overrides the actual path.
#     Job names: srl-1k, srl-2k, srl-5k, srl-10k, srl-14k, srl-22k
#
#     ALREADY RUNNING — left in place for reference; uncomment to resubmit.
# ============================================================================
# SFTED_STEPS=(1000 2000 5000 10000 14000 22000)
# INTERLEAVE_SFT=0
# INTERLEAVE_PPO=50000
#
# for PRETRAIN in "${SFTED_STEPS[@]}"; do
#     SUFFIX=$(step_to_suffix $PRETRAIN)
#     SFTED_DIR="${SFTED_GSM_BASE}/OLMo2-1B_step${PRETRAIN}_interleave_twoloader_n32_sft_50000_ppo_0_gsm/hf_model/step2000"
#
#     echo "Submitting srl-${SUFFIX} (RL on SFTed-2k of pretrain ${PRETRAIN}) — 24h"
#     sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
#         --job-name="srl-${SUFFIX}" --time=24:00:00 \
#         --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${PRETRAIN},OLMO_CHECKPOINT=${SFTED_DIR},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},EXP_TAG=sfted,EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_gsm" \
#         ${GSM_SLURM}
#     COUNT=$((COUNT + 1))
#     sleep 1
# done

# ============================================================================
# §3 — Curriculum SFT->RL on 10k pretrain checkpoint — 24h, 2 GPU
#     Two schedules. Both SFT and RL data are openmathinstruct2_gsm8k (dedupe).
#     Job names: curd-10k (debug), curn-10k (novel)
# ============================================================================
STEP=10000
SUFFIX=$(step_to_suffix $STEP)

# Schedule 1 — Debug curriculum: sft 250 / ppo 50, increment 20
#     ALREADY RUNNING — left in place for reference; uncomment to resubmit.
# echo "Submitting curd-${SUFFIX} (curriculum debug: sft 250 / ppo 50, ±20) — 24h"
# sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
#     --job-name="curd-${SUFFIX}" --time=24:00:00 \
#     --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},NUM_SFT_STEPS=250,NUM_PPO_STEPS=50,RL_INCREMENT=20,SFT_DECREMENT=20,LR_SCALE=1.0 \
#     ${CURRICULUM_SLURM}
# COUNT=$((COUNT + 1))
# sleep 1

# Schedule 2 — Novel curriculum: sft 800 / ppo 200, increment 50
echo "Submitting curn-${SUFFIX} (curriculum novel: sft 800 / ppo 200, ±50) — 24h"
sbatch --account=${SLURM_ACCOUNT} --mail-type=FAIL --mail-user=${MAIL_USER} \
    --job-name="curn-${SUFFIX}" --time=24:00:00 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},NUM_SFT_STEPS=800,NUM_PPO_STEPS=200,RL_INCREMENT=50,SFT_DECREMENT=50,LR_SCALE=1.0 \
    ${CURRICULUM_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# §4 — MATH RL on 60BMATH pretrain checkpoints — 72h, 4 GPU
#     RL data: openmathinstruct2 (MATH dedupe, hardcoded in math inner script).
#     SFT_DATA_DIR set for completeness but unused (NUM_SFT_STEPS=0).
#     Run the chmod block ONCE before submitting (see plan file).
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
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
