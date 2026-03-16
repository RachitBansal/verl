#!/bin/bash
#SBATCH --job-name=verl-grpo-olmo2-openmath
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Parse command line arguments for environment variables
ENV_VAR_NAME=""
ENV_VAR_VALUE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --step_num) STEP_NUM="$2"; export STEP_NUM; shift 2 ;;
    --lr_scale) LR_SCALE="$2"; export LR_SCALE; shift 2 ;;
    --num_sft_steps) num_sft_steps="$2"; export num_sft_steps; shift 2 ;;
    --num_ppo_steps) num_ppo_steps="$2"; export num_ppo_steps; shift 2 ;;
    *) break ;;
  esac
done

while [[ $# -gt 0 ]]; do
  case $1 in
    --env-var)
      if [[ $2 == *"="* ]]; then
        # Format: --env-var NAME=VALUE
        ENV_VAR_NAME="${2%%=*}"
        ENV_VAR_VALUE="${2#*=}"
      else
        # Format: --env-var VALUE (use default name)
        ENV_VAR_NAME="CUSTOM_VAR"
        ENV_VAR_VALUE="$2"
      fi
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

cd /n/home05/sqin/rl_pretrain/verl

# Run the main training script
echo "Running training script with remaining arguments: $@"
bash examples/grpo_trainer/test_sft_rl.sh "$@"

echo "Training completed!"

