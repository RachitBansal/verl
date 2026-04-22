#!/bin/bash
#SBATCH --job-name=paravg-test
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Usage:
#   sbatch --export=ALL,VERL_DIR=...,CONDA_ENV=...,STEP_NUM=10000,RL_LR=1e-6,SFT_LR=4e-5 \
#       examples/grpo_trainer/run_olmo2-1b_gsm_parallel_avg_slurm.sh

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "STEP_NUM: ${STEP_NUM:-10000}"
echo "RL_LR: ${RL_LR:-1e-6}"
echo "SFT_LR: ${SFT_LR:-4e-5}"
echo ""

set -x

echo "Starting SLURM job on $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE, Tasks: $SLURM_NTASKS_PER_NODE"

cd ${VERL_DIR}

# Run the parallel_avg SFT+RL training script
bash examples/grpo_trainer/run_olmo2-1b_gsm_parallel_avg.sh "$@"

echo "Training completed!"
