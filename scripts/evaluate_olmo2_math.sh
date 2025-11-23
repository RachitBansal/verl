#!/bin/bash
#
# Evaluation script for OLMo-2 model on GSM-8K and MATH datasets
# Supports few-shot evaluation and top-k sampling strategies
#
# Usage:
#   bash evaluate_olmo2_math.sh
#
# Author: Generated for OLMo-2 evaluation
# Date: $(date +%Y-%m-%d)

set -e  # Exit on error
set -u  # Exit on undefined variable

#############################################
# CONFIGURATION
#############################################

# Evaluation Control
EVAL_GSM8K=true   # Set to true to evaluate on GSM8K
EVAL_MATH=false    # Set to true to evaluate on MATH

# Model Configuration
MODEL_PATH=/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1-50B/step22000-hf
MODEL_NAME="1B-step22000"
N_SAMPLES=1


# Hardware Configuration
NNODES=1
N_GPUS_PER_NODE=1
TENSOR_PARALLEL_SIZE=1  # Increase if model doesn't fit in single GPU
GPU_MEMORY_UTIL=0.85

# Dataset Configuration
BASE_DIR="/n/netscratch/dam_lab/Lab/sqin/rl_pretrain"
DATA_DIR="${BASE_DIR}/data"
GSM8K_DIR="${DATA_DIR}/gsm8k"
MATH_DIR="${DATA_DIR}/math"

# Evaluation Configuration
N_SHOT=8  # Number of few-shot examples (matching interleaved-rl)
TEMPERATURE=0.0  # 0.0 for greedy, >0 for sampling
TOP_P=0.95
TOP_K=-1  # -1 means no top-k filtering (matching interleaved-rl)

# Generation Configuration
BATCH_SIZE=256
MAX_PROMPT_LENGTH=2048  # Increased to accommodate 8-shot examples (~750-850 tokens)
MAX_RESPONSE_LENGTH=1024

# Output Configuration
OUTPUT_DIR="${BASE_DIR}/eval_results/${MODEL_NAME}-${N_SHOT}shot-${N_SAMPLES}samples-temp${TEMPERATURE}"
mkdir -p "${OUTPUT_DIR}"

# Logging
LOG_FILE="${OUTPUT_DIR}/evaluation.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Wandb Configuration
WANDB_PROJECT="interleaved-rl"
WANDB_ENTITY="harvardml"
WANDB_RUN_NAME="${MODEL_NAME}_${N_SHOT}shot_${N_SAMPLES}samples_temp${TEMPERATURE}"

echo "================================================"
echo "OLMo-2 Math Evaluation Script"
echo "================================================"
echo "Model: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "N-shot: ${N_SHOT}"
echo "N-samples (top-k): ${N_SAMPLES}"
echo "Temperature: ${TEMPERATURE}"
echo ""
echo "Wandb Configuration:"
echo "  Project: ${WANDB_PROJECT}"
echo "  Entity: ${WANDB_ENTITY}"
echo "  Run Name: ${WANDB_RUN_NAME}"
echo "================================================"
echo ""

#############################################
# STEP 0: Verify Model Exists
#############################################

echo "[Step 0/5] Verifying model checkpoint..."
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: Model config.json not found in: ${MODEL_PATH}"
    echo "This should be a HuggingFace format model directory."
    exit 1
fi

echo "✓ Model checkpoint verified"
echo ""

#############################################
# STEP 1: Prepare Datasets
#############################################

echo "[Step 1/5] Preparing datasets..."

# GSM-8K Dataset
if [ "${EVAL_GSM8K}" = true ]; then
    if [ ! -f "${GSM8K_DIR}/test.parquet" ]; then
        echo "Preparing GSM-8K dataset..."
        mkdir -p "${GSM8K_DIR}"
        cd examples/data_preprocess
        python3 gsm8k.py --local_save_dir "${GSM8K_DIR}"
        cd - > /dev/null
        echo "✓ GSM-8K dataset prepared"
    else
        echo "✓ GSM-8K dataset already exists"
    fi
else
    echo "⊘ GSM-8K evaluation disabled"
fi

# MATH Dataset
if [ "${EVAL_MATH}" = true ]; then
    if [ ! -f "${MATH_DIR}/test.parquet" ]; then
        echo "Preparing MATH dataset..."
        mkdir -p "${MATH_DIR}"
        cd examples/data_preprocess
        python3 math_dataset.py --local_save_dir "${MATH_DIR}"
        cd - > /dev/null
        echo "✓ MATH dataset prepared"
    else
        echo "✓ MATH dataset already exists"
    fi
else
    echo "⊘ MATH evaluation disabled"
fi

echo ""

#############################################
# STEP 2: Apply Few-Shot Examples (if needed)
#############################################

if [ "${N_SHOT}" -gt 0 ]; then
    echo "[Step 2/5] Applying ${N_SHOT}-shot examples..."

    # Create few-shot datasets
    if [ "${EVAL_GSM8K}" = true ]; then
        GSM8K_FEWSHOT="${GSM8K_DIR}/test_${N_SHOT}shot.parquet"
        python3 scripts/create_fewshot_dataset.py \
            --input_file "${GSM8K_DIR}/test.parquet" \
            --train_file "${GSM8K_DIR}/train.parquet" \
            --output_file "${GSM8K_FEWSHOT}" \
            --n_shot "${N_SHOT}" \
            --dataset_type "gsm8k"
        GSM8K_TEST_FILE="${GSM8K_FEWSHOT}"
    fi

    if [ "${EVAL_MATH}" = true ]; then
        MATH_FEWSHOT="${MATH_DIR}/test_${N_SHOT}shot.parquet"
        python3 scripts/create_fewshot_dataset.py \
            --input_file "${MATH_DIR}/test.parquet" \
            --train_file "${MATH_DIR}/train.parquet" \
            --output_file "${MATH_FEWSHOT}" \
            --n_shot "${N_SHOT}" \
            --dataset_type "math"
        MATH_TEST_FILE="${MATH_FEWSHOT}"
    fi

    echo "✓ Few-shot datasets created"
else
    echo "[Step 2/5] Skipping few-shot (0-shot evaluation)..."
    if [ "${EVAL_GSM8K}" = true ]; then
        GSM8K_TEST_FILE="${GSM8K_DIR}/test.parquet"
    fi
    if [ "${EVAL_MATH}" = true ]; then
        MATH_TEST_FILE="${MATH_DIR}/test.parquet"
    fi
fi

echo ""

#############################################
# STEP 3: Generate Responses - GSM-8K
#############################################

if [ "${EVAL_GSM8K}" = true ]; then
    echo "[Step 3/5] Generating responses on GSM-8K..."

    GSM8K_OUTPUT="${OUTPUT_DIR}/gsm8k_predictions.parquet"

    if [ ! -f "${GSM8K_OUTPUT}" ]; then
        python3 -m verl.trainer.main_generation \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
            trainer.device=cuda \
            data.path="${GSM8K_TEST_FILE}" \
            data.prompt_key=prompt \
            data.n_samples="${N_SAMPLES}" \
            data.output_path="${GSM8K_OUTPUT}" \
            data.batch_size="${BATCH_SIZE}" \
            model.path="${MODEL_PATH}" \
            +model.trust_remote_code=True \
            rollout.name=vllm \
            rollout.temperature="${TEMPERATURE}" \
            rollout.top_k="${TOP_K}" \
            rollout.top_p="${TOP_P}" \
            rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
            rollout.response_length="${MAX_RESPONSE_LENGTH}" \
            rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
            +rollout.pipeline_model_parallel_size=1 \
            rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
            rollout.dtype=bfloat16 \
            rollout.enforce_eager=True \
            ray_kwargs.ray_init.num_cpus=16

        echo "✓ GSM-8K responses generated: ${GSM8K_OUTPUT}"
    else
        echo "✓ GSM-8K responses already exist: ${GSM8K_OUTPUT}"
    fi
    echo ""
else
    echo "[Step 3/5] Skipping GSM-8K generation (disabled)"
    echo ""
fi

#############################################
# STEP 4: Generate Responses - MATH
#############################################

if [ "${EVAL_MATH}" = true ]; then
    echo "[Step 4/5] Generating responses on MATH dataset..."

    MATH_OUTPUT="${OUTPUT_DIR}/math_predictions.parquet"

    if [ ! -f "${MATH_OUTPUT}" ]; then
        python3 -m verl.trainer.main_generation \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
            trainer.device=cuda \
            data.path="${MATH_TEST_FILE}" \
            data.prompt_key=prompt \
            data.n_samples="${N_SAMPLES}" \
            data.output_path="${MATH_OUTPUT}" \
            data.batch_size="${BATCH_SIZE}" \
            model.path="${MODEL_PATH}" \
            +model.trust_remote_code=True \
            rollout.name=vllm \
            rollout.temperature="${TEMPERATURE}" \
            rollout.top_k="${TOP_K}" \
            rollout.top_p="${TOP_P}" \
            rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
            rollout.response_length="${MAX_RESPONSE_LENGTH}" \
            rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
            +rollout.pipeline_model_parallel_size=1 \
            rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
            rollout.dtype=bfloat16 \
            rollout.enforce_eager=True \
            ray_kwargs.ray_init.num_cpus=16

        echo "✓ MATH responses generated: ${MATH_OUTPUT}"
    else
        echo "✓ MATH responses already exist: ${MATH_OUTPUT}"
    fi
    echo ""
else
    echo "[Step 4/5] Skipping MATH generation (disabled)"
    echo ""
fi

# ############################################
# STEP 5: Evaluate Results
# ############################################

echo "[Step 5/5] Evaluating results..."

# Evaluate GSM-8K
if [ "${EVAL_GSM8K}" = true ]; then
    echo ""
    echo "--- GSM-8K Results ---"
    python3 -m verl.trainer.main_eval \
        data.path="${GSM8K_OUTPUT}" \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +logger='["wandb"]' \
        +wandb.project="${WANDB_PROJECT}" \
        +wandb.entity="${WANDB_ENTITY}" \
        +wandb.run_name="${WANDB_RUN_NAME}_gsm8k" \
        +wandb.tags='["gsm8k","evaluation"]' \
        ray_kwargs.ray_init.num_cpus=16 | tee "${OUTPUT_DIR}/gsm8k_results.txt"
fi

# Evaluate MATH
if [ "${EVAL_MATH}" = true ]; then
    echo ""
    echo "--- MATH Results ---"
    python3 -m verl.trainer.main_eval \
        data.path="${MATH_OUTPUT}" \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +logger='["wandb"]' \
        +wandb.project="${WANDB_PROJECT}" \
        +wandb.entity="${WANDB_ENTITY}" \
        +wandb.run_name="${WANDB_RUN_NAME}_math" \
        +wandb.tags='["math","evaluation"]' \
        ray_kwargs.ray_init.num_cpus=16 | tee "${OUTPUT_DIR}/math_results.txt"
fi

# If N_SAMPLES > 1, also evaluate with majority voting
if [ "${N_SAMPLES}" -gt 1 ]; then
    echo ""
    echo "--- Top-${N_SAMPLES} Majority Voting Evaluation ---"

    if [ "${EVAL_GSM8K}" = true ]; then
        echo "GSM-8K Majority Voting:"
        python3 scripts/evaluate_majority_voting.py \
            --input_file "${GSM8K_OUTPUT}" \
            --dataset_type "gsm8k" | tee "${OUTPUT_DIR}/gsm8k_majority_results.txt"
        echo ""
    fi

    if [ "${EVAL_MATH}" = true ]; then
        echo "MATH Majority Voting:"
        python3 scripts/evaluate_majority_voting.py \
            --input_file "${MATH_OUTPUT}" \
            --dataset_type "math" | tee "${OUTPUT_DIR}/math_majority_results.txt"
    fi
fi

#############################################
# SUMMARY
#############################################

echo ""
echo "================================================"
echo "Evaluation Complete!"
echo "================================================"
echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  N-shot: ${N_SHOT}"
echo "  N-samples: ${N_SAMPLES}"
echo "  Temperature: ${TEMPERATURE}"
echo ""
echo "Output files:"
echo "  Log: ${LOG_FILE}"
echo "  GSM-8K predictions: ${GSM8K_OUTPUT}"
echo "  GSM-8K results: ${OUTPUT_DIR}/gsm8k_results.txt"
echo "  MATH predictions: ${MATH_OUTPUT}"
echo "  MATH results: ${OUTPUT_DIR}/math_results.txt"
if [ "${N_SAMPLES}" -gt 1 ]; then
    echo "  GSM-8K majority voting: ${OUTPUT_DIR}/gsm8k_majority_results.txt"
    echo "  MATH majority voting: ${OUTPUT_DIR}/math_majority_results.txt"
fi
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/gsm8k_results.txt"
echo "  cat ${OUTPUT_DIR}/math_results.txt"
echo ""
echo "To view results on Wandb:"
echo "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "================================================"
