#!/usr/bin/env bash
# Clara's launcher (2026-04-29): SFTed-RL resume + new 4B SFT-only sweep.
#
# Sections:
#   1. Resume RL on already-SFTed 1B ckpts   (srl-*,    4 A100 x 48h)
#        SFT: none in this run (NUM_SFT_STEPS=0); base ckpt was pre-SFTed on GSM.
#        RL data: openmathinstruct2_gsm8k (hardcoded in gsm interleave inner script).
#        Resumes existing experiments matching:
#          OLMo2-1B_step{N}sfted_interleave_twoloader_n32_sft_0_ppo_50000_gsm
#        for N in {1000, 2000, 5000}. Trainer auto-picks up the latest checkpoint
#        from default_local_dir; OLMO_CHECKPOINT is only used at step 0.
#   2. SFT-only on 4B pretrain ckpts         (m4bs-*,   2 H100 x 48h)
#        SFT data: openmathinstruct2 (MATH dedupe, NOT rmath).
#        RL: none (NUM_PPO_STEPS=0).  lr=5e-4.  save_freq=500/test_freq=250.
#        MODEL_NAME=OLMo2-4B → exp name includes "4B":
#          OLMo2-4B_step{5000,14000}_interleave_twoloader_n32_sft_50000_ppo_0_math
#
# Comment out steps in STEPS_*=( ... ) arrays to split work between collaborators.
#
# Usage: bash examples/grpo_trainer/launch_experiments_clara_04_29.sh

set -eo pipefail

# ============================================================================
# USER CONFIG — change these for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_barak_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
PARTITION="kempner"   # A100 partition (a100-40gb); use kempner_h100 for H100
# ============================================================================

# Inner SLURM dispatchers
GSM_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_gsm_interleave_slurm.sh"
MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"

# Data directories
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

# Checkpoint directories
CKPT_4B="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-4B-stage1-50B"

# SFTed checkpoint base (each pretrain step has hf_model/step2000 inside)
SFTED_GSM_BASE="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

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
# §1 — Resume RL on already-SFTed 1B checkpoints — 48h, 4 A100
#     STEP_NUM is used as a label only; OLMO_CHECKPOINT overrides the actual path.
#     save_freq=50 / test_freq=25.
#     Job names: srl-1k, srl-2k, srl-5k
# ============================================================================
SFTED_STEPS=(1000 2000 5000)
INTERLEAVE_SFT=0
INTERLEAVE_PPO=50000

for PRETRAIN in "${SFTED_STEPS[@]}"; do
    SUFFIX=$(step_to_suffix $PRETRAIN)
    SFTED_DIR="${SFTED_GSM_BASE}/OLMo2-1B_step${PRETRAIN}_interleave_twoloader_n32_sft_50000_ppo_0_gsm/hf_model/step2000"

    echo "Submitting srl-${SUFFIX} (resume RL on SFTed-2k of pretrain ${PRETRAIN}) — 48h, 4 A100"
    sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="srl-${SUFFIX}" --time=48:00:00 \
        --gpus-per-node=4 --cpus-per-task=64 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${PRETRAIN},OLMO_CHECKPOINT=${SFTED_DIR},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},SAVE_FREQ=50,TEST_FREQ=25,EXP_TAG=sfted,EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_gsm" \
        ${GSM_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

# ============================================================================
# §2 — SFT-only on 4B pretrain checkpoints (steps 5k, 14k) — 48h, 2 H100
#     Mirrors Rachit's 60BMATH SFT-only setup but on the 4B-stage1-50B model.
#     SFT data: openmathinstruct2 (MATH dedupe, NOT rmath).
#     50000 SFT steps, 0 PPO steps. lr=5e-4. save_freq=500 / test_freq=250.
#     MODEL_NAME=OLMo2-4B → exp name like
#       OLMo2-4B_step5000_interleave_twoloader_n32_sft_50000_ppo_0_math
#     Job names: m4bs-5k, m4bs-14k
# ============================================================================
STEPS_4B_SFT=(5000 14000)   # comment out steps already submitted
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
