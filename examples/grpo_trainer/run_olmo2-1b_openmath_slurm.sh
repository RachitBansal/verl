#!/bin/bash
#SBATCH --job-name=rl-22k
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#SBATCH --constraint=h100

module purge
module load Mambaforge
module load cuda cudnn
mamba activate openrlhf

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo ""

set -x

echo "Starting SLURM job on $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE, Tasks: $SLURM_NTASKS_PER_NODE"

# cd /n/home05/sqin/rl_pretrain/verl

# Get STEP_NUM from first argument or default to empty
STEP_NUM=${1:-}
if [ -z "$STEP_NUM" ]; then
    echo "Error: STEP_NUM is required as first argument"
    echo "Usage: sbatch run_olmo2-1b_openmath_slurm.sh <STEP_NUM>"
    exit 1
fi

export STEP_NUM
echo "Running with STEP_NUM=$STEP_NUM"

# Run the main training script
bash /n/home05/sqin/rl_pretrain/verl/examples/grpo_trainer/run_olmo2-1b_openmath.sh "${@:2}"

echo "Training completed!"

