#!/bin/bash
# Submit RL-only jobs starting from SFT-ed checkpoints
# Uses 2 GPUs per job, 6 jobs total (12 GPUs), kempner_barak_lab account

set -e

EXPERIMENTS_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"
SLURM_SCRIPT="examples/grpo_trainer/run_olmo2-1b_openmath_gsm_interleave_slurm.sh"
VERL_DIR="/n/home05/sqin/rl_pretrain/verl"
CONDA_ENV="/n/holylabs/dam_lab/Lab/brachit/envs/bin/activate"

# step -> hf_model subdir mapping
declare -A HF_CKPT_STEPS
HF_CKPT_STEPS[1000]=step9500
HF_CKPT_STEPS[3000]=step9500
HF_CKPT_STEPS[5000]=step9500
HF_CKPT_STEPS[6000]=step9500
HF_CKPT_STEPS[10000]=step9500
HF_CKPT_STEPS[22000]=step9500

for STEP in 1000 3000 5000 6000 10000 22000; do
    HF_STEP=${HF_CKPT_STEPS[$STEP]}
    CKPT_PATH="${EXPERIMENTS_DIR}/OLMo2-1B_step${STEP}_interleave_twoloader_n32_sft_50000_ppo_0_rgsm/hf_model/${HF_STEP}"

    echo "Submitting step=${STEP} from checkpoint: ${CKPT_PATH}"

    sbatch \
        --job-name="sfted-rl-${STEP}" \
        --account=kempner_barak_lab \
        --partition=kempner_h100 \
        --gpus-per-node=2 \
        --time=12:00:00 \
        --export=ALL,VERL_DIR=${VERL_DIR},CONDA_ENV=${CONDA_ENV},STEP_NUM=${STEP},OLMO_CHECKPOINT=${CKPT_PATH},NUM_SFT_STEPS=0,NUM_PPO_STEPS=50000,EXP_TAG=sfted,EXP_SUFFIX="interleave_twoloader_n32_sft_0_ppo_50000_rgsm" \
        ${SLURM_SCRIPT}
done

echo "All 6 jobs submitted."
