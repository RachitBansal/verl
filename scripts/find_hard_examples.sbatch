#!/bin/bash
#SBATCH --job-name=rl-filtering
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/find-hard-examples-%j.out
#SBATCH --error=logs/find-hard-examples-%j.out

# ============================================================================
# Find Hard Examples from OpenMathInstruct2 Dataset
# 
# This script identifies examples where OLMo-2-0425-1B has low accuracy (~1/64)
# when generating 128 samples. These "hard" examples are valuable for 
# adaptive rollout experiments.
#
# Author: Sunny + Claude
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module purge
module load Mambaforge
module load cuda cudnn

# Activate environment
source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# Print job info
echo "============================================================"
echo "Finding Hard Examples from OpenMathInstruct2"
echo "============================================================"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
# echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"
echo ""

set -x

# ============================================================================
# Configuration - can be overridden via environment variables
# ============================================================================
# Model (default: HuggingFace OLMo-2-0425-1B)
MODEL_PATH="${MODEL_PATH:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo-2-0425-1B"}"

# Data
DATA_PATH="${DATA_PATH:-/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k/train_gsm8k.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k_hard}"

# Generation settings
N_SAMPLES="${N_SAMPLES:-128}"         # Samples per example
TEMPERATURE="${TEMPERATURE:-0.6}"     # Sampling temperature
TOP_P="${TOP_P:-0.95}"                # Top-p sampling

# Hard example selection
TARGET_ACCURACY="${TARGET_ACCURACY:-0.015625}"  # 1/64
TOLERANCE="${TOLERANCE:-0.01}"                   # ±3%
N_HARD_EXAMPLES="${N_HARD_EXAMPLES:-1000}"
MAX_SCAN="${MAX_SCAN:-50000}"

# GPU settings
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  N samples per example: $N_SAMPLES"
echo "  Target accuracy: $TARGET_ACCURACY"
echo "  Tolerance: ±$TOLERANCE"
echo "  N hard examples to find: $N_HARD_EXAMPLES"
echo ""

# Change to verl directory
cd /n/netscratch/dam_lab/Lab/brachit/rollouts/verl

# vLLM multiprocessing fix for CUDA
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Run the script
python3 scripts/find_hard_examples.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --target_accuracy "$TARGET_ACCURACY" \
    --tolerance "$TOLERANCE" \
    --n_hard_examples "$N_HARD_EXAMPLES" \
    --max_examples_to_scan "$MAX_SCAN" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --save_all_scores \
    "$@"

echo ""
echo "============================================================"
echo "Job completed at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

