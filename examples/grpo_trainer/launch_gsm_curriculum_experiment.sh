#!/usr/bin/env bash
# Launch a single curriculum SFT->RL experiment on OLMo2-1B / GSM8K.
# Starts with many SFT steps and few RL steps, shifting toward pure RL each round.
#
# Usage: bash examples/grpo_trainer/launch_gsm_curriculum_experiment.sh

set -euo pipefail

# ============================================================================
# USER CONFIG — change these 3 lines for your setup
# ============================================================================
SLURM_ACCOUNT="kempner_sham_lab"
VERL_DIR="/n/home03/cmohri/team_verl"
CONDA_ENV="/n/home03/cmohri/venvs/verl_env/bin/activate"
# ============================================================================

CURRICULUM_SLURM="examples/grpo_trainer/run_olmo2-1b_openmath_curriculum_slurm.sh"

mkdir -p logs

# Curriculum schedule: 200 SFT / 1 RL, shift by 1 each round until pure RL.
STEP=1000
NUM_SFT_STEPS=200
NUM_PPO_STEPS=1
RL_INCREMENT=1
SFT_DECREMENT=1
LR_SCALE=1.0

echo "Submitting curriculum (step ${STEP}, sft=${NUM_SFT_STEPS}, ppo=${NUM_PPO_STEPS}, rl_inc=${RL_INCREMENT}, sft_dec=${SFT_DECREMENT}) — 48h"
sbatch --account=${SLURM_ACCOUNT} --job-name="curric-${STEP}" --time=48:00:00 \
    --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},NUM_SFT_STEPS=${NUM_SFT_STEPS},NUM_PPO_STEPS=${NUM_PPO_STEPS},RL_INCREMENT=${RL_INCREMENT},SFT_DECREMENT=${SFT_DECREMENT},LR_SCALE=${LR_SCALE} \
    ${CURRICULUM_SLURM}

echo ""
echo "Submitted 1 job."
