#!/usr/bin/env bash
# Clara's launcher (2026-05-03).
#
# Sections:
#   1. Requeue m60s-10k (originally Rachit's)            (m60s-10k, 4 A100 x 48h)
#        [ALREADY SUBMITTED — section commented out]
#        SFT-only on OLMo2-1B-60BMATH step 10000, MATH dedupe data, lr=5e-4.
#        4 A100s required to match the saved checkpoint sharding at:
#          /n/netscratch/dam_lab/Everyone/rl_pretrain/experiments/
#            OLMo2-1B-60BMATH_step10000_interleave_twoloader_n32_sft_50000_ppo_0_math
#        Trainer auto-picks up the latest checkpoint from default_local_dir.
#   2. parallel_avg with matched-low LR (RL=1e-6, SFT=1e-7) (pavg-low, 4 H100 x 48h)
#        [ALREADY SUBMITTED — section commented out]
#        Mirror of §1 of launch_gsm_sft_experiments_clara.sh but with SFT_LR=1e-7.
#        Exp name: OLMo2-1B_step10000_parallel_avg_n32_rl1e-6_sft1e-7_smbatch
#   3. RL on already-SFTed 60BMATH ckpts                  (srlm-*,   2 H100 x 48h)
#        SFT data: none (NUM_SFT_STEPS=0; interleave trick — pure RL).
#        RL data:  openmathinstruct2 (MATH dedupe, hardcoded in math interleave inner script).
#        Base ckpt: hf_model/step10000 if present, else hf_model/step9500
#          (standardized SFT-progress snapshot inside each SFT-only dir
#          OLMo2-1B-60BMATH_step{N}_interleave_twoloader_n32_sft_50000_ppo_0_math).
#        Pretrain steps: 5k, 17k, 22k, 28k (14k omitted — no hf_model yet).
#        Exp names:
#          OLMo2-1B-60BMATH_step{N}sfted_interleave_twoloader_n32_sft_0_ppo_50000_math
#
# Comment out steps in STEPS_*=( ... ) arrays / individual sbatch blocks to split
# work between collaborators.
#
# Usage: bash examples/grpo_trainer/launch_experiments_clara_05_03.sh

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
PARAVG_SLURM="examples/grpo_trainer/run_olmo2-1b_gsm_parallel_avg_slurm.sh"

# Data directories
MATH_DEDUP="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

# Checkpoint directories
CKPT_60B="/n/netscratch/barak_lab/Everyone/sqin/olmo/checkpoints/OLMo2-1B-stage1-60B"

# Base dir for already-SFTed 60BMATH experiments (each has hf_model/step{N})
SFTED_60BMATH_BASE="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

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
# §1 — Requeue m60s-10k (60BMATH SFT-only on step 10k) — 48h, 4 A100
#     Picks up the existing checkpoint (resume); GPU count must equal the
#     world size used at save time (4 A100s here).
#     Exp name: OLMo2-1B-60BMATH_step10000_interleave_twoloader_n32_sft_50000_ppo_0_math
#     Job name: m60s-10k
#
#     ALREADY SUBMITTED — uncomment to resubmit.
# ============================================================================
# STEPS_60BMATH_SFT=(10000)
# SFTONLY_SFT=50000
# SFTONLY_PPO=0
#
# for STEP in "${STEPS_60BMATH_SFT[@]}"; do
#     SUFFIX=$(step_to_suffix $STEP)
#
#     echo "Submitting m60s-${SUFFIX} (resume 60BMATH SFT-only on MATH dedupe, step ${STEP}) — 48h, 4 A100"
#     sbatch --account=${SLURM_ACCOUNT} --partition=kempner \
#         --mail-type=FAIL --mail-user=${MAIL_USER} \
#         --job-name="m60s-${SUFFIX}" --time=48:00:00 \
#         --gpus-per-node=4 --cpus-per-task=64 \
#         --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},CHECKPOINT_DIR=${CKPT_60B},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${SFTONLY_SFT},NUM_PPO_STEPS=${SFTONLY_PPO},SAVE_FREQ=500,TEST_FREQ=250,RL_LR=5e-4,EXP_SUFFIX="interleave_twoloader_n32_sft_${SFTONLY_SFT}_ppo_${SFTONLY_PPO}_math" \
#         ${MATH_SLURM}
#     COUNT=$((COUNT + 1))
#     sleep 1
# done

# ============================================================================
# §2 — parallel_avg, matched-low LR (RL=1e-6, SFT=1e-7) — 48h, 4 H100
#     Mirror of §1 of launch_gsm_sft_experiments_clara.sh but SFT_LR lowered
#     from 1e-6 to 1e-7. Same batch fix (SCALE_BATCH=False → _smbatch tag).
#     Exp name: OLMo2-1B_step10000_parallel_avg_n32_rl1e-6_sft1e-7_smbatch
#     Job name: pavg-low
#
#     ALREADY SUBMITTED — uncomment to resubmit.
# ============================================================================
# echo "Submitting pavg-low (parallel_avg, RL=1e-6 / SFT=1e-7) — 48h, 4 H100"
# sbatch --account=${SLURM_ACCOUNT} --partition=kempner_h100 \
#     --mail-type=FAIL --mail-user=${MAIL_USER} \
#     --job-name="pavg-low" --time=48:00:00 \
#     --gpus-per-node=4 --cpus-per-task=96 \
#     --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=10000,RL_LR=1e-6,SFT_LR=1e-7,SCALE_BATCH=False,SAVE_FREQ=50,TEST_FREQ=25 \
#     ${PARAVG_SLURM}
# COUNT=$((COUNT + 1))
# sleep 1

# ============================================================================
# §3 — RL on already-SFTed 60BMATH checkpoints — 48h, 2 H100
#     Interleave trick (NUM_SFT_STEPS=0, NUM_PPO_STEPS=50000) so de-facto pure RL.
#     STEP_NUM=PRETRAIN labels the experiment; OLMO_CHECKPOINT points at the
#     hf_model/step10000 (or step9500 fallback) inside each SFT-only experiment dir.
#     EXP_TAG=sfted → exp name OLMo2-1B-60BMATH_step{N}sfted_..._math.
#     Job names: srlm-5k, srlm-17k, srlm-22k, srlm-28k
#     (14k omitted — no hf_model snapshot yet; add back when ready)
# ============================================================================
SFTED_60BMATH_STEPS=(5000 17000 22000 28000)   # 14k omitted: no hf_model snapshot yet
INTERLEAVE_SFT=0
INTERLEAVE_PPO=50000

for PRETRAIN in "${SFTED_60BMATH_STEPS[@]}"; do
    SUFFIX=$(step_to_suffix $PRETRAIN)
    EXP_DIR="${SFTED_60BMATH_BASE}/OLMo2-1B-60BMATH_step${PRETRAIN}_interleave_twoloader_n32_sft_50000_ppo_0_math"

    # Standardize SFT snapshot: prefer step10000, fall back to step9500.
    SFT_HF=""
    for CAND in step10000 step9500; do
        if [[ -d "${EXP_DIR}/hf_model/${CAND}" ]]; then
            SFT_HF="${EXP_DIR}/hf_model/${CAND}"
            break
        fi
    done

    if [[ -z "${SFT_HF}" ]]; then
        echo "SKIP srlm-${SUFFIX}: neither hf_model/step10000 nor step9500 in ${EXP_DIR}"
        continue
    fi

    echo "Submitting srlm-${SUFFIX} (RL on SFTed 60BMATH pretrain ${PRETRAIN}, hf=$(basename ${SFT_HF})) — 48h, 2 H100"
    sbatch --account=${SLURM_ACCOUNT} --partition=kempner_h100 \
        --mail-type=FAIL --mail-user=${MAIL_USER} \
        --job-name="srlm-${SUFFIX}" --time=48:00:00 \
        --gpus-per-node=2 --cpus-per-task=48 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${PRETRAIN},OLMO_CHECKPOINT=${SFT_HF},MODEL_NAME=OLMo2-1B-60BMATH,SFT_DATA_DIR=${MATH_DEDUP},NUM_SFT_STEPS=${INTERLEAVE_SFT},NUM_PPO_STEPS=${INTERLEAVE_PPO},SAVE_FREQ=50,TEST_FREQ=25,EXP_TAG=sfted,EXP_SUFFIX="interleave_twoloader_n32_sft_${INTERLEAVE_SFT}_ppo_${INTERLEAVE_PPO}_math" \
        ${MATH_SLURM}
    COUNT=$((COUNT + 1))
    sleep 1
done

echo ""
echo "Submitted ${COUNT} jobs total."
