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
#SBATCH --error=logs/slurm-%j.err

# Parse command line arguments for environment variables
ENV_VAR_NAME=""
ENV_VAR_VALUE=""
STEP_NUM=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --exp_num)
      ENV_VAR_NAME="exp_num"
      ENV_VAR_VALUE="$2"
      shift 2
      ;;
    --step_num)
      STEP_NUM="$2"
      shift 2
      ;;
    *)
      # Pass through other arguments
      break
      ;;
  esac
done

# Set the environment variable if specified
if [[ -n "$ENV_VAR_NAME" ]]; then
  export $ENV_VAR_NAME="$ENV_VAR_VALUE"
  echo "Set environment variable: $ENV_VAR_NAME=$ENV_VAR_VALUE"
else
  # Default exp_num to 1 if not specified
  export exp_num=1
  echo "Set default exp_num=1"
fi

# Export STEP_NUM if specified
if [[ -n "$STEP_NUM" ]]; then
  export STEP_NUM="$STEP_NUM"
  echo "Set STEP_NUM: $STEP_NUM"
fi

# module purge
# module load Mambaforge
# module load cuda cudnn
# mamba activate openrlhf

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

VERL_DIR="${VERL_DIR:-/n/home05/sqin/rl_pretrain/verl}"
cd "${VERL_DIR}"

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
bash "${VERL_DIR}/examples/grpo_trainer/run_olmo2-1b_openmath.sh" "${@:2}"

echo "Training completed!"
