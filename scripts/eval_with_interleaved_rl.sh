#!/bin/bash
#
# Use interleaved-rl's evaluation script to test on first 200 samples
# This ensures we're using identical generation method to interleaved-rl
#

set -e

MODEL_PATH="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1-50B/step22000-hf"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/sqin/rl_pretrain/eval_results/interleaved-rl-full"

echo "================================================"
echo "Running interleaved-rl evaluation on all samples"
echo "================================================"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

cd /n/home05/sqin/rl_pretrain/interleaved-rl/interleaved-rl

python scripts/eval_math_models.py \
    --model "${MODEL_PATH}" \
    --datasets gsm8k \
    --experiment math_evaluation \
    --variant olmo2_1b \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 32 \
    --gsm8k_shots 8 \
    --math_shots 4 \
    --gsm8k_samples 1319 \
    --math_samples 0 \
    --temperature 0.6 \
    --pass_k 1 \
    --device cuda \
    --dtype bfloat16

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To compare with your verl results, check:"
echo "  ${OUTPUT_DIR}/step22000-hf_math/gsm8k_detailed_results.json"
