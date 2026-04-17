#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --output=logs/lm-eval-%j.out
#SBATCH --error=logs/lm-eval-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=kempner
#SBATCH --account=kempner_dam_lab

set -e
set -u

# Activate environment
export PATH="/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf/bin:$PATH"

# Usage: sbatch lm_eval_harness/run_eval.sh /path/to/hf_checkpoint
MODEL_PATH="${1:?Usage: sbatch run_eval.sh <model_path>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"

mkdir -p "${BASE_DIR}/logs"

echo "================================================"
echo "LM Eval Harness - Base Capability Evaluation"
echo "Model: ${MODEL_PATH}"
echo "Started at: $(date)"
echo "================================================"
echo ""

cd "${BASE_DIR}"

python lm_eval_harness/run_eval.py --model_path "${MODEL_PATH}"

echo ""
echo "================================================"
echo "Completed at: $(date)"
echo "================================================"
