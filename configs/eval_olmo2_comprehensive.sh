#!/bin/bash
#
# Comprehensive evaluation configuration for OLMo-2
# Run multiple evaluation strategies and compare results
#
# Usage: bash configs/eval_olmo2_comprehensive.sh
#

set -e

# Base configuration
export MODEL_PATH="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1-50B/step14000-hf"
export MODEL_NAME="olmo2-1b-step14000"
export OUTPUT_BASE="${HOME}/data/eval_results/${MODEL_NAME}_comprehensive_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_BASE}"

echo "=========================================="
echo "Comprehensive OLMo-2 Evaluation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_BASE}"
echo ""

# Function to run evaluation with specific config
run_eval() {
    local name=$1
    local n_shot=$2
    local n_samples=$3
    local temperature=$4
    local top_p=$5
    
    echo ""
    echo ">>> Running: ${name}"
    echo "    N-shot: ${n_shot}, N-samples: ${n_samples}, Temp: ${temperature}"
    
    local output_dir="${OUTPUT_BASE}/${name}"
    mkdir -p "${output_dir}"
    
    # Update configuration in the main script
    cp evaluate_olmo2_math.sh "${output_dir}/eval_script.sh"
    
    sed -i \
        -e "s|^MODEL_PATH=.*|MODEL_PATH=\"${MODEL_PATH}\"|" \
        -e "s|^MODEL_NAME=.*|MODEL_NAME=\"${MODEL_NAME}_${name}\"|" \
        -e "s|^OUTPUT_DIR=.*|OUTPUT_DIR=\"${output_dir}\"|" \
        -e "s|^N_SHOT=.*|N_SHOT=${n_shot}|" \
        -e "s|^N_SAMPLES=.*|N_SAMPLES=${n_samples}|" \
        -e "s|^TEMPERATURE=.*|TEMPERATURE=${temperature}|" \
        -e "s|^TOP_P=.*|TOP_P=${top_p}|" \
        "${output_dir}/eval_script.sh"
    
    # Run evaluation
    cd "${output_dir}"
    bash eval_script.sh
    cd - > /dev/null
    
    echo "    ✓ Complete: ${output_dir}"
}

# 1. Baseline: 0-shot greedy
run_eval "0shot_greedy" 0 1 0.0 1.0

# 2. Few-shot: 4-shot greedy
run_eval "4shot_greedy" 4 1 0.0 1.0

# 3. Few-shot: 8-shot greedy
run_eval "8shot_greedy" 8 1 0.0 1.0

# 4. Top-8 sampling with majority vote
run_eval "0shot_top8" 0 8 0.7 0.95

# 5. Combined: 4-shot + top-8
run_eval "4shot_top8" 4 8 0.7 0.95

# Generate summary report
echo ""
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

SUMMARY_FILE="${OUTPUT_BASE}/SUMMARY.txt"

{
    echo "=========================================="
    echo "OLMo-2 Comprehensive Evaluation Summary"
    echo "=========================================="
    echo "Model: ${MODEL_PATH}"
    echo "Date: $(date)"
    echo ""
    echo "=========================================="
    echo "Results"
    echo "=========================================="
    echo ""
    
    for eval_dir in "${OUTPUT_BASE}"/*/; do
        eval_name=$(basename "${eval_dir}")
        
        echo ">>> ${eval_name}"
        echo ""
        
        if [ -f "${eval_dir}/gsm8k_results.txt" ]; then
            echo "GSM-8K:"
            grep -A 1 "test_score" "${eval_dir}/gsm8k_results.txt" 2>/dev/null || echo "  No results found"
        fi
        
        if [ -f "${eval_dir}/math_results.txt" ]; then
            echo "MATH:"
            grep -A 1 "test_score" "${eval_dir}/math_results.txt" 2>/dev/null || echo "  No results found"
        fi
        
        if [ -f "${eval_dir}/gsm8k_majority_results.txt" ]; then
            echo "GSM-8K Majority Voting:"
            grep "Accuracy\|Pass@" "${eval_dir}/gsm8k_majority_results.txt" 2>/dev/null || echo "  No results found"
        fi
        
        if [ -f "${eval_dir}/math_majority_results.txt" ]; then
            echo "MATH Majority Voting:"
            grep "Accuracy\|Pass@" "${eval_dir}/math_majority_results.txt" 2>/dev/null || echo "  No results found"
        fi
        
        echo ""
        echo "----------------------------------------"
        echo ""
    done
    
    echo ""
    echo "=========================================="
    echo "Output Directory Structure"
    echo "=========================================="
    find "${OUTPUT_BASE}" -type f -name "*.txt" -o -name "*.parquet" | head -20
    echo ""
    echo "Full results in: ${OUTPUT_BASE}"
    
} | tee "${SUMMARY_FILE}"

echo ""
echo "=========================================="
echo "Comprehensive Evaluation Complete!"
echo "=========================================="
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "To view results:"
echo "  cat ${SUMMARY_FILE}"
echo "  ls -lh ${OUTPUT_BASE}/*/*.txt"
echo "=========================================="


