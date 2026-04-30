#!/usr/bin/env bash
# Rachit's launcher (2026-04-29): resume 60BMATH direct-RL on A100s.
#
# Sections:
#   1. Resume MATH RL on 60BMATH pretrain ckpt (m60rl-5k, 4 A100 x 72h)
#        SFT: none (NUM_SFT_STEPS=0; SFT_DATA_DIR set for completeness but unused).
#        RL data: openmathinstruct2 (MATH dedupe, hardcoded in math interleave inner script).
#        Resumes existing experiment:
#          OLMo2-1B-60BMATH_step5000_interleave_twoloader_n32_sft_0_ppo_50000_rmath
#        Trainer auto-picks up the latest checkpoint from default_local_dir.
#   2. SFT-only on 60BMATH pretrain ckpts (m60s-14k, m60s-10k; 4 A100 x 72h)
#        SFT data: openmathinstruct2 (MATH dedupe, NOT rmath).
#        RL: none (NUM_PPO_STEPS=0).  lr=5e-4.  save_freq=500/test_freq=250.
#
# Comment out steps in STEPS_*=( ... ) arrays to split work between collaborators.
#
# Usage: bash examples/grpo_trainer/launch_experiments_rachit_04_29.sh

set -eo pipefail

# ============================================================================
# USER CONFIG — change these for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_dam_lab"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"
MAIL_USER="tqin@g.harvard.edu"
PARTITION="kempner"   # A100 partition (a100-40gb); use kempner_h100 for H100
# ============================================================================

# Inner SLURM dispatchers
MATH_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_math_interleave_slurm.sh"

# Data directories
MATH_DUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_duplicated"
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

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
# §1 — Resume MATH RL on 60BMATH pretrain checkpoint (step 5k) — 72h, 4 A100
#     save_freq=50 / test_freq=25.
#     Job names: m60rl-5k
# ============================================================================
STEPS_60BMATH_RL=(5000)
INTERLEAVE_SFT=0
INTERLEAVE_PPO=50000

for STEP in "${STEPS_60BMATH_RL[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting m60rl-${SUFFIX} (resume 60BMATH MATH RL, step ${STEP}) — 72h, 4 A100"
    sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="m60rl-${SUFFIX}" --time=72:00:00 \
        --gpus-per-node=4 --cpus-per-task=64 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DUP},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},SAVE_FREQ=50,TEST_FREQ=25,EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_rmath" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

# ============================================================================
# §2 — SFT-only on 60BMATH pretrain checkpoints (steps 14k, 10k) — 72h, 4 A100
#     Trains on MATH dedupe (openmathinstruct2), NOT rmath (the duplicated set).
#     50000 SFT steps, 0 PPO steps. lr=5e-4. save_freq=500 / test_freq=250.
#     Job names: m60s-14k, m60s-10k
#     Exp names: OLMo2-1B-60BMATH_step{14000,10000}_interleave_twoloader_n32_sft_50000_ppo_0_math
# ============================================================================
STEPS_60BMATH_SFT=(14000 10000)   # comment out steps already submitted
SFTONLY_SFT=50000
SFTONLY_PPO=0

for STEP in "${STEPS_60BMATH_SFT[@]}"; do
    SUFFIX=$(step_to_suffix $STEP)

    echo "Submitting m60s-${SUFFIX} (60BMATH SFT-only on MATH dedupe, step ${STEP}) — 72h, 4 A100"
    sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="m60s-${SUFFIX}" --time=72:00:00 \
        --gpus-per-node=4 --cpus-per-task=64 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${SFTONLY_SFT},NUM_PPO_STEPS=${SFTONLY_PPO},SAVE_FREQ=500,TEST_FREQ=250,RL_LR=5e-4,EXP_SUFFIX="interleave_twoloader_n32_sft_${SFTONLY_SFT}_ppo_${SFTONLY_PPO}_math" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
