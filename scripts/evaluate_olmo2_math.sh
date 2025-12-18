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

# # ---- Activate your prebuilt env
# source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# # Use the env's Python explicitly everywhere:
# PYBIN=/n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/python
# echo "Python: $PYBIN"

#############################################
# CONFIGURATION
#############################################

source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# Evaluation Control (defaults; can be overridden by args)
EVAL_GSM8K_DEFAULT=true    # arg4
EVAL_MATH_DEFAULT=false    # arg5
OPENMATHINSTRUCT2=false    # Keep disabled unless manually toggled

# Dataset Preparation Control
OPENMATHINSTRUCT2_N_SAMPLES=5000  # Number of samples to randomly select

# Model Configuration defaults (can be overridden by args)
MODEL_PATH_DEFAULT=/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step5000-hf  # arg1
MODEL_NAME_DEFAULT="1B-step5000"  # arg2
N_SAMPLES_DEFAULT=1  # arg3

# Positional arguments from launcher:
#   $1 = MODEL_PATH
#   $2 = MODEL_NAME
#   $3 = N_SAMPLES
#   $4 = EVAL_GSM8K (true/false)
#   $5 = EVAL_MATH (true/false)
MODEL_PATH="${1:-${MODEL_PATH_DEFAULT}}"
MODEL_NAME="${2:-${MODEL_NAME_DEFAULT}}"
N_SAMPLES="${3:-${N_SAMPLES_DEFAULT}}"
EVAL_GSM8K="${4:-${EVAL_GSM8K_DEFAULT}}"
EVAL_MATH="${5:-${EVAL_MATH_DEFAULT}}"


# Hardware Configuration
NNODES=1
N_GPUS_PER_NODE=1
TENSOR_PARALLEL_SIZE=1  # Increase if model doesn't fit in single GPU
GPU_MEMORY_UTIL=0.85

# Dataset Configuration
BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain"
DATA_DIR="${BASE_DIR}/data"
CACHE_DIR="/n/netscratch/dam_lab/Lab/sqin/cache"  # HuggingFace datasets cache - change it per user 
GSM8K_DIR="${DATA_DIR}/gsm8k"
MATH_DIR="${DATA_DIR}/math"
OPENMATHINSTRUCT2_DIR="${DATA_DIR}/openmathinstruct2"

# Evaluation Configuration
N_SHOT=1  # Number of few-shot examples
TEMPERATURE=0.0  # 0.0 for greedy, >0 for sampling
TOP_P=0.95
TOP_K=-1  # -1 means no top-k filtering 

if [ "${N_SAMPLES}" -gt 1 ]; then
    TEMPERATURE=0.6
fi

# Generation Configuration
BATCH_SIZE=256
MAX_PROMPT_LENGTH=2048  # Increased to accommodate 8-shot examples (~750-850 tokens)
MAX_RESPONSE_LENGTH=1024

# Output Configuration
OUTPUT_DIR="${BASE_DIR}/eval_results/${MODEL_NAME}-${N_SHOT}shot-${N_SAMPLES}samples-temp${TEMPERATURE}"
mkdir -p "${OUTPUT_DIR}"

# Set HuggingFace cache directory
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}"
mkdir -p "${CACHE_DIR}"

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
echo "Cache Directory: ${CACHE_DIR}"
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

echo "[Step 0/6] Verifying model checkpoint..."
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

echo "[Step 1/6] Preparing datasets..."

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

# OpenMathInstruct-2 Dataset
if [ "${OPENMATHINSTRUCT2}" = true ]; then
    if [ ! -f "${OPENMATHINSTRUCT2_DIR}/train_subset.parquet" ]; then
        echo "Preparing OpenMathInstruct-2 dataset (${OPENMATHINSTRUCT2_N_SAMPLES} samples)..."
        mkdir -p "${OPENMATHINSTRUCT2_DIR}"
        cd examples/data_preprocess
        python3 openmathinstruct2.py \
            --local_save_dir "${OPENMATHINSTRUCT2_DIR}" \
            --cache_dir "${CACHE_DIR}" \
            --n_samples "${OPENMATHINSTRUCT2_N_SAMPLES}"
        cd - > /dev/null
        echo "✓ OpenMathInstruct-2 dataset prepared"
    else
        echo "✓ OpenMathInstruct-2 dataset already exists"
    fi
else
    echo "⊘ OpenMathInstruct-2 preparation disabled"
fi

echo ""

#############################################
# STEP 2: Apply Few-Shot Examples (if needed)
#############################################

if [ "${N_SHOT}" -gt 0 ]; then
    echo "[Step 2/6] Applying ${N_SHOT}-shot examples..."

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

    if [ "${OPENMATHINSTRUCT2}" = true ] && [ -f "${OPENMATHINSTRUCT2_DIR}/train_subset.parquet" ]; then
        OPENMATHINSTRUCT2_FEWSHOT="${OPENMATHINSTRUCT2_DIR}/train_subset_${N_SHOT}shot.parquet"
        python3 scripts/create_fewshot_dataset.py \
            --input_file "${OPENMATHINSTRUCT2_DIR}/train_subset.parquet" \
            --output_file "${OPENMATHINSTRUCT2_FEWSHOT}" \
            --n_shot "${N_SHOT}" \
            --dataset_type "openmathinstruct2"
        OPENMATHINSTRUCT2_FILE="${OPENMATHINSTRUCT2_FEWSHOT}"
    fi

    echo "✓ Few-shot datasets created"
else
    echo "[Step 2/6] Skipping few-shot (0-shot evaluation)..."
    if [ "${EVAL_GSM8K}" = true ]; then
        GSM8K_TEST_FILE="${GSM8K_DIR}/test.parquet"
    fi
    if [ "${EVAL_MATH}" = true ]; then
        MATH_TEST_FILE="${MATH_DIR}/test.parquet"
    fi
    if [ "${OPENMATHINSTRUCT2}" = true ]; then
        OPENMATHINSTRUCT2_FILE="${OPENMATHINSTRUCT2_DIR}/train_subset.parquet"
    fi
fi

echo ""

#############################################
# STEP 3: Generate Responses - GSM-8K
#############################################

if [ "${EVAL_GSM8K}" = true ]; then
    echo "[Step 3/6] Generating responses on GSM-8K..."

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
    echo "[Step 3/6] Skipping GSM-8K generation (disabled)"
    echo ""
fi

#############################################
# STEP 4: Generate Responses - MATH
#############################################

if [ "${EVAL_MATH}" = true ]; then
    echo "[Step 4/6] Generating responses on MATH dataset..."

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
    echo "[Step 4/6] Skipping MATH generation (disabled)"
    echo ""
fi

#############################################
# STEP 5: Generate Responses - OpenMathInstruct-2
#############################################

if [ "${OPENMATHINSTRUCT2}" = true ] && [ -n "${OPENMATHINSTRUCT2_FILE}" ]; then
    echo "[Step 5/6] Generating responses on OpenMathInstruct-2 dataset..."

    OPENMATHINSTRUCT2_OUTPUT="${OUTPUT_DIR}/openmathinstruct2_predictions.parquet"

    if [ ! -f "${OPENMATHINSTRUCT2_OUTPUT}" ]; then
        python3 -m verl.trainer.main_generation \
            trainer.nnodes="${NNODES}" \
            trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
            trainer.device=cuda \
            data.path="${OPENMATHINSTRUCT2_FILE}" \
            data.prompt_key=prompt \
            data.n_samples="${N_SAMPLES}" \
            data.output_path="${OPENMATHINSTRUCT2_OUTPUT}" \
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

        echo "✓ OpenMathInstruct-2 responses generated: ${OPENMATHINSTRUCT2_OUTPUT}"
    else
        echo "✓ OpenMathInstruct-2 responses already exist: ${OPENMATHINSTRUCT2_OUTPUT}"
    fi
    echo ""
else
    echo "[Step 5/6] Skipping OpenMathInstruct-2 generation (disabled)"
    echo ""
fi

# ############################################
# STEP 6: Evaluate Results
# ############################################

echo "[Step 6/6] Evaluating results..."

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

# Evaluate OpenMathInstruct-2
if [ "${OPENMATHINSTRUCT2}" = true ] && [ -n "${OPENMATHINSTRUCT2_OUTPUT}" ]; then
    echo ""
    echo "--- OpenMathInstruct-2 Results ---"
    python3 -m verl.trainer.main_eval \
        data.path="${OPENMATHINSTRUCT2_OUTPUT}" \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
        custom_reward_function.name=compute_score_data_source \
        +logger='["wandb"]' \
        +wandb.project="${WANDB_PROJECT}" \
        +wandb.entity="${WANDB_ENTITY}" \
        +wandb.run_name="${WANDB_RUN_NAME}_openmathinstruct2" \
        +wandb.tags='["openmathinstruct2","evaluation"]' \
        ray_kwargs.ray_init.num_cpus=16 | tee "${OUTPUT_DIR}/openmathinstruct2_results.txt"
fi

# If N_SAMPLES > 1, also evaluate with majority voting
if [ "${N_SAMPLES}" -gt 1 ]; then
    echo ""
    echo "--- Top-${N_SAMPLES} Majority Voting Evaluation ---"

    if [ "${EVAL_GSM8K}" = true ]; then
        echo "GSM-8K Majority Voting:"
        python3 -m scripts.evaluate_majority_voting \
            --input_file "${GSM8K_OUTPUT}" \
            --dataset_type "gsm8k" | tee "${OUTPUT_DIR}/gsm8k_majority_results.txt"
        echo ""
    fi

    if [ "${EVAL_MATH}" = true ]; then
        echo "MATH Majority Voting:"
        python3 -m scripts.evaluate_majority_voting \
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
if [ "${EVAL_GSM8K}" = true ]; then
    echo "  GSM-8K predictions: ${GSM8K_OUTPUT}"
    echo "  GSM-8K results: ${OUTPUT_DIR}/gsm8k_results.txt"
fi
if [ "${EVAL_MATH}" = true ]; then
    echo "  MATH predictions: ${MATH_OUTPUT}"
    echo "  MATH results: ${OUTPUT_DIR}/math_results.txt"
fi
if [ "${OPENMATHINSTRUCT2}" = true ] && [ -n "${OPENMATHINSTRUCT2_OUTPUT}" ]; then
    echo "  OpenMathInstruct-2 predictions: ${OPENMATHINSTRUCT2_OUTPUT}"
    echo "  OpenMathInstruct-2 results: ${OUTPUT_DIR}/openmathinstruct2_results.txt"
fi
if [ "${N_SAMPLES}" -gt 1 ]; then
    if [ "${EVAL_GSM8K}" = true ]; then
        echo "  GSM-8K majority voting: ${OUTPUT_DIR}/gsm8k_majority_results.txt"
    fi
    if [ "${EVAL_MATH}" = true ]; then
        echo "  MATH majority voting: ${OUTPUT_DIR}/math_majority_results.txt"
    fi
fi
echo ""
echo "To view results:"
if [ "${EVAL_GSM8K}" = true ]; then
    echo "  cat ${OUTPUT_DIR}/gsm8k_results.txt"
fi
if [ "${EVAL_MATH}" = true ]; then
    echo "  cat ${OUTPUT_DIR}/math_results.txt"
fi
if [ "${OPENMATHINSTRUCT2}" = true ] && [ -n "${OPENMATHINSTRUCT2_OUTPUT}" ]; then
    echo "  cat ${OUTPUT_DIR}/openmathinstruct2_results.txt"
fi
echo ""
echo "To view results on Wandb:"
echo "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "================================================"
