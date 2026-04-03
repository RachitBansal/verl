#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

# Usage:
#   sbatch --job-name=gsm-1k \
#       --export=ALL,STEP_NUM=1000,SFT_DATA_DIR=...,NUM_SFT_STEPS=5000,NUM_PPO_STEPS=0,EXP_SUFFIX=... \
#       examples/grpo_trainer/run_olmo2-1b_openmath_gsm_interleave_slurm.sh

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "STEP_NUM: $STEP_NUM"
echo "SFT_DATA_DIR: $SFT_DATA_DIR"
echo "NUM_SFT_STEPS: $NUM_SFT_STEPS"
echo "NUM_PPO_STEPS: $NUM_PPO_STEPS"
echo "EXP_SUFFIX: $EXP_SUFFIX"
echo ""

set -x

echo "Starting SLURM job on $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE, Tasks: $SLURM_NTASKS_PER_NODE"

cd ${VERL_DIR}

# Run the interleaved SFT+RL training script
bash examples/grpo_trainer/run_olmo2-1b_openmath_gsm_interleave.sh "$@"

echo "Training completed!"
