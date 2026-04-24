#!/usr/bin/env bash
# Clara's experiment launcher — parallel_avg / combined sweep on GSM8K.
# All four runs use the new batch fix: sft_config.scale_batch_by_rollout_n=False
# (SFT mini-batch matched in count to RL mini-batch, so each SFT mini is
# rollout.n × smaller than the old behavior). All on 50B-stage1 step10000.
#
# Sections (each is one job; comment out to skip):
#   1. parallel_avg, matched LR (RL=1e-6, SFT=1e-6)        (pavg-mat,  4 GPU x 48h)
#   2. parallel_avg, asymmetric LR (RL=1e-6, SFT=4e-5)     (pavg-asy,  4 GPU x 48h)
#   3. combined, token-sum SFT loss, 1:1 weights, lr=1e-6  (comb-tsm,  4 GPU x 48h)
#   4. combined, token-mean SFT loss, 1:1 weights, lr=1e-6 (comb-tmn,  4 GPU x 48h)
#
# Data (all four):
#   SFT data: openmathinstruct2_gsm8k  (dedupe)   ← two separate data loaders,
#   RL data:  openmathinstruct2_gsm8k  (dedupe)     same underlying dir
#   parallel_avg inner script hardcodes both to gsm8k dedupe.
#   combined inner script: we pass SFT_DATA_DIR explicitly so SFT loader
#   reads from the dedupe set instead of the duplicated default.
#
# Usage: bash examples/grpo_trainer/launch_gsm_sft_experiments_clara.sh

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
PARAVG_SLURM="examples/grpo_trainer/run_olmo2-1b_gsm_parallel_avg_slurm.sh"
COMBINED_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_combined_slurm.sh"

# Common
STEP=10000
GPU=4
TIME="48:00:00"
PARTITION="kempner_h100"
SAVE_FREQ=50
TEST_FREQ=25
SCALE_BATCH=False        # the new fix: SFT mini = RL mini / rollout.n
GSM_DEDUPE="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k"

mkdir -p logs
COUNT=0

# ============================================================================
# §1 — parallel_avg, matched LR (RL=1e-6, SFT=1e-6) — 48h, 4 GPU
#     Baseline for "two balanced models averaged" under the new batch fix.
#     Exp name suffix: _smbatch (added automatically when SCALE_BATCH=False).
#     Job name: pavg-mat
# ============================================================================
echo "Submitting pavg-mat (parallel_avg, matched LR rl=1e-6/sft=1e-6) — 48h, 4 GPU"
sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
    --mail-type=FAIL --mail-user=${MAIL_USER} \
    --job-name="pavg-mat" --time=${TIME} \
    --gpus-per-node=${GPU} --cpus-per-task=96 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},RL_LR=1e-6,SFT_LR=1e-6,SCALE_BATCH=${SCALE_BATCH},SAVE_FREQ=${SAVE_FREQ},TEST_FREQ=${TEST_FREQ} \
    ${PARAVG_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# §2 — parallel_avg, asymmetric LR (RL=1e-6, SFT=4e-5) — 48h, 4 GPU
#     Original failing config — control to confirm SFT-dominated behavior persists
#     under the new batch fix.
#     Job name: pavg-asy
# ============================================================================
echo "Submitting pavg-asy (parallel_avg, asym LR rl=1e-6/sft=4e-5) — 48h, 4 GPU"
sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
    --mail-type=FAIL --mail-user=${MAIL_USER} \
    --job-name="pavg-asy" --time=${TIME} \
    --gpus-per-node=${GPU} --cpus-per-task=96 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},RL_LR=1e-6,SFT_LR=4e-5,SCALE_BATCH=${SCALE_BATCH},SAVE_FREQ=${SAVE_FREQ},TEST_FREQ=${TEST_FREQ} \
    ${PARAVG_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# §3 — combined, token-sum SFT loss, 1:1 weights, lr=1e-6 — 48h, 4 GPU
#     Hypothesis: with matched 1:1 loss weights and Adam's shared v, SFT's
#     token-sum gradient still dominates. Exp-name suffix: _token-sum_smbatch.
#     Job name: comb-tsm
# ============================================================================
echo "Submitting comb-tsm (combined, token-sum SFT, 1:1, lr=1e-6) — 48h, 4 GPU"
sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
    --mail-type=FAIL --mail-user=${MAIL_USER} \
    --job-name="comb-tsm" --time=${TIME} \
    --gpus-per-node=${GPU} --cpus-per-task=96 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},RL_LR=1e-6,LR_SCALE=1.0,RL_LOSS_WEIGHT=1.0,SFT_LOSS_WEIGHT=1.0,SFT_LOSS_AGG_MODE=token-sum,SCALE_BATCH=${SCALE_BATCH},SFT_DATA_DIR=${GSM_DEDUPE} \
    ${COMBINED_SLURM}
COUNT=$((COUNT + 1))
sleep 1

# ============================================================================
# §4 — combined, token-mean SFT loss, 1:1 weights, lr=1e-6 — 48h, 4 GPU
#     Hypothesis: token-mean on SFT balances the shared Adam v and lets RL
#     actually contribute. Exp-name suffix: _token-mean_smbatch.
#     Job name: comb-tmn
# ============================================================================
echo "Submitting comb-tmn (combined, token-mean SFT, 1:1, lr=1e-6) — 48h, 4 GPU"
sbatch --account=${SLURM_ACCOUNT} --partition=${PARTITION} \
    --mail-type=FAIL --mail-user=${MAIL_USER} \
    --job-name="comb-tmn" --time=${TIME} \
    --gpus-per-node=${GPU} --cpus-per-task=96 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},RL_LR=1e-6,LR_SCALE=1.0,RL_LOSS_WEIGHT=1.0,SFT_LOSS_WEIGHT=1.0,SFT_LOSS_AGG_MODE=token-mean,SCALE_BATCH=${SCALE_BATCH},SFT_DATA_DIR=${GSM_DEDUPE} \
    ${COMBINED_SLURM}
COUNT=$((COUNT + 1))
sleep 1

echo ""
echo "Submitted ${COUNT} jobs total."
