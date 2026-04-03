#!/bin/bash
#SBATCH --job-name=1k-combined
#SBATCH --account=kempner_barak_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Usage:
#   sbatch --export=ALL,STEP_NUM=1000,RL_LOSS_WEIGHT=1.0,SFT_LOSS_WEIGHT=1.0,LR_SCALE=1.0 \
#       examples/grpo_trainer/run_olmo2-1b_openmath_combined_slurm.sh

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --step_num)         STEP_NUM="$2";         export STEP_NUM;         shift 2 ;;  # pretrain checkpoint step number
    --lr_scale)         LR_SCALE="$2";         export LR_SCALE;         shift 2 ;;  # SFT learning rate multiplier
    --rl_loss_weight)   RL_LOSS_WEIGHT="$2";   export RL_LOSS_WEIGHT;   shift 2 ;;  # weight on RL loss term
    --sft_loss_weight)  SFT_LOSS_WEIGHT="$2";  export SFT_LOSS_WEIGHT;  shift 2 ;;  # weight on SFT loss term
    *) break ;;
  esac
done

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "STEP_NUM: $STEP_NUM"
echo "LR_SCALE: $LR_SCALE"
echo "RL_LOSS_WEIGHT: $RL_LOSS_WEIGHT"
echo "SFT_LOSS_WEIGHT: $SFT_LOSS_WEIGHT"
echo ""

set -x

echo "Starting SLURM job on $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE, Tasks: $SLURM_NTASKS_PER_NODE"

cd /n/home05/sqin/rl_pretrain/verl

# Run the combined SFT+RL training script
bash examples/grpo_trainer/run_olmo2-1b_openmath_combined.sh "$@"

echo "Training completed!"
