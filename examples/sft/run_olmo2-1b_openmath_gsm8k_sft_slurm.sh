#!/bin/bash
#SBATCH --job-name=verl-sft-olmo2-openmath
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=00:30:00
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
    --stepnum)
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

cd /n/home05/sqin/rl_pretrain/verl

# Run the SFT training script
echo "Running SFT script with remaining arguments: $@"
bash examples/sft_trainer/run_olmo2-1b_openmath_gsm8k_sft.sh "$@"

echo "SFT training completed!"