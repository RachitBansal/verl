#!/bin/bash
#
# Evaluation script for OLMo-2 model on GSM-8K and MATH datasets
# Adapted for user permissions and verified environment
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==========================================================
# 1. SETUP ENVIRONMENT (From your working config)
# ==========================================================

# Source the venv
source /n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/activate

# Use the env's Python explicitly:
PYBIN=/n/netscratch/sham_lab/Everyone/cmohri/venvs/verl/bin/python
echo "Python: $PYBIN"

# Export API keys and settings
export WANDB_API_KEY="cdc354a166dcf9cbcd8b1eed1fd2fe89d03f8b90"
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::ResourceWarning,ignore::FutureWarning"
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1

# ---- Fast local caches (Fixes the dam_lab/Lab permission issue) ----
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache-$USER}
export HF_HOME=${HF_HOME:-/tmp/hf-home-$USER}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" || true

# ==========================================================
# 2. CONFIGURATION
# ==========================================================

# Evaluation Control
EVAL_GSM8K=true
EVAL_MATH=false
OPENMATHINSTRUCT2=false
OPENMATHINSTRUCT2_N_SAMPLES=5000

# Model Configuration
# NOTE: Ensure you have READ access to this path. If not, copy the checkpoint to your scratch.
MODEL_PATH="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step14000-hf"
MODEL_NAME="1B-step14000"
N_SAMPLES=1

# Hardware
NNODES=1
N_GPUS_PER_NODE=1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.85

# --- PATH CORRECTIONS ---
# We point BASE_DIR to YOUR scratch space so you can write results/logs
USER_SCRATCH="/n/netscratch/sham_lab/Everyone/cmohri" 
BASE_DIR="${USER_SCRATCH}/rl_pretrain_experiments" 

DATA_DIR="${BASE_DIR}/data"
# Use the local cache we defined above
CACHE_DIR="${HF_HOME}"

GSM8K_DIR="${DATA_DIR}/gsm8k"
MATH_DIR="${DATA_DIR}/math"
OPENMATHINSTRUCT2_DIR="${DATA_DIR}/openmathinstruct2"

# Evaluation Params
N_SHOT=8
TEMPERATURE=0.0
TOP_P=0.95
TOP_K=-1

# Generation Params
BATCH_SIZE=256
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1024

# Output Configuration
OUTPUT_DIR="${BASE_DIR}/eval_results/${MODEL_NAME}-${N_SHOT}shot-${N_SAMPLES}samples-temp${TEMPERATURE}"
mkdir -p "${OUTPUT_DIR}"

# Logging
LOG_FILE="${OUTPUT_DIR}/evaluation.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Wandb
WANDB_PROJECT="interleaved-rl"
WANDB_ENTITY="harvardml"
WANDB_RUN_NAME="${MODEL_NAME}_${N_SHOT}shot_${N_SAMPLES}samples_temp${TEMPERATURE}"

echo "================================================"
echo "OLMo-2 Math Evaluation Script (Corrected)"
echo "================================================"
echo "Model: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Cache Directory: ${CACHE_DIR}"
echo "================================================"

#############################################
# STEP 0: Verify Model Exists
#############################################

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path does not exist or is not accessible: ${MODEL_PATH}"
    exit 1
fi

#############################################
# STEP 1: Prepare Datasets
#############################################

# NOTE: All 'python3' calls replaced with '$PYBIN'

echo "[Step 1/6] Preparing datasets..."

if [ "${EVAL_GSM8K}" = true ]; then
    if [ ! -f "${GSM8K_DIR}/test.parquet" ]; then
        mkdir -p "${GSM8K_DIR}"
        cd examples/data_preprocess
        $PYBIN gsm8k.py --local_save_dir "${GSM8K_DIR}"
        cd - > /dev/null
    fi
fi

if [ "${EVAL_MATH}" = true ]; then
    if [ ! -f "${MATH_DIR}/test.parquet" ]; then
        mkdir -p "${MATH_DIR}"
        cd examples/data_preprocess
        $PYBIN math_dataset.py --local_save_dir "${MATH_DIR}"
        cd - > /dev/null
    fi
fi

if [ "${OPENMATHINSTRUCT2}" = true ]; then
    if [ ! -f "${OPENMATHINSTRUCT2_DIR}/train_subset.parquet" ]; then
        mkdir -p "${OPENMATHINSTRUCT2_DIR}"
        cd examples/data_preprocess
        $PYBIN openmathinstruct2.py \
            --local_save_dir "${OPENMATHINSTRUCT2_DIR}" \
            --cache_dir "${CACHE_DIR}" \
            --n_samples "${OPENMATHINSTRUCT2_N_SAMPLES}"
        cd - > /dev/null
    fi
fi

#############################################
# STEP 2: Apply Few-Shot
#############################################

if [ "${N_SHOT}" -gt 0 ]; then
    echo "[Step 2/6] Applying ${N_SHOT}-shot examples..."
    
    if [ "${EVAL_GSM8K}" = true ]; then
        GSM8K_FEWSHOT="${GSM8K_DIR}/test_${N_SHOT}shot.parquet"
        $PYBIN scripts/create_fewshot_dataset.py \
            --input_file "${GSM8K_DIR}/test.parquet" \
            --train_file "${GSM8K_DIR}/train.parquet" \
            --output_file "${GSM8K_FEWSHOT}" \
            --n_shot "${N_SHOT}" \
            --dataset_type "gsm8k"
        GSM8K_TEST_FILE="${GSM8K_FEWSHOT}"
    fi

    if [ "${EVAL_MATH}" = true ]; then
        MATH_FEWSHOT="${MATH_DIR}/test_${N_SHOT}shot.parquet"
        $PYBIN scripts/create_fewshot_dataset.py \
            --input_file "${MATH_DIR}/test.parquet" \
            --train_file "${MATH_DIR}/train.parquet" \
            --output_file "${MATH_FEWSHOT}" \
            --n_shot "${N_SHOT}" \
            --dataset_type "math"
        MATH_TEST_FILE="${MATH_FEWSHOT}"
    fi

    if [ "${OPENMATHINSTRUCT2}" = true ] && [ -f "${OPENMATHINSTRUCT2_DIR}/train_subset.parquet" ]; then
        OPENMATHINSTRUCT2_FEWSHOT="${OPENMATHINSTRUCT2_DIR}/train_subset_${N_SHOT}shot.parquet"
        $PYBIN scripts/create_fewshot_dataset.py \
            --input_file "${OPENMATHINSTRUCT2_DIR}/train_subset.parquet" \
            --output_file "${OPENMATHINSTRUCT2_FEWSHOT}" \
            --n_shot "${N_SHOT}" \
            --dataset_type "openmathinstruct2"
        OPENMATHINSTRUCT2_FILE="${OPENMATHINSTRUCT2_FEWSHOT}"
    fi
else
    # Logic for 0-shot (omitted for brevity, assume same as original logic)
    if [ "${EVAL_GSM8K}" = true ]; then GSM8K_TEST_FILE="${GSM8K_DIR}/test.parquet"; fi
    if [ "${EVAL_MATH}" = true ]; then MATH_TEST_FILE="${MATH_DIR}/test.parquet"; fi
    if [ "${OPENMATHINSTRUCT2}" = true ]; then OPENMATHINSTRUCT2_FILE="${OPENMATHINSTRUCT2_DIR}/train_subset.parquet"; fi
fi

#############################################
# STEP 3-5: Generate Responses
#############################################

# GSM8K
if [ "${EVAL_GSM8K}" = true ]; then
    GSM8K_OUTPUT="${OUTPUT_DIR}/gsm8k_predictions.parquet"
    echo "Eval gsm8k"
    if [ ! -f "${GSM8K_OUTPUT}" ]; then
        $PYBIN -m verl.trainer.main_generation \
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
            +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
            +actor_rollout_ref.rollout.dtype=bfloat16 \
            +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
            rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
            +rollout.pipeline_model_parallel_size=1 \
            rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
            rollout.dtype=bfloat16 \
            rollout.enforce_eager=True \
            ray_kwargs.ray_init.num_cpus=16
    fi
fi

# MATH
if [ "${EVAL_MATH}" = true ]; then
    MATH_OUTPUT="${OUTPUT_DIR}/math_predictions.parquet"
    if [ ! -f "${MATH_OUTPUT}" ]; then
        $PYBIN -m verl.trainer.main_generation \
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
            +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
            +actor_rollout_ref.rollout.dtype=bfloat16 \
            +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
            rollout.response_length="${MAX_RESPONSE_LENGTH}" \
            rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
            +rollout.pipeline_model_parallel_size=1 \
            rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
            rollout.dtype=bfloat16 \
            rollout.enforce_eager=True \
            ray_kwargs.ray_init.num_cpus=16
    fi
fi

# OPENMATHINSTRUCT2
if [ "${OPENMATHINSTRUCT2}" = true ] && [ -n "${OPENMATHINSTRUCT2_FILE}" ]; then
    OPENMATHINSTRUCT2_OUTPUT="${OUTPUT_DIR}/openmathinstruct2_predictions.parquet"
    if [ ! -f "${OPENMATHINSTRUCT2_OUTPUT}" ]; then
        $PYBIN -m verl.trainer.main_generation \
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
            +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
            +actor_rollout_ref.rollout.dtype=bfloat16 \
            +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
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
    fi
fi

#############################################
# STEP 6: Evaluate Results
#############################################

# GSM8K Eval
if [ "${EVAL_GSM8K}" = true ]; then
    $PYBIN -m verl.trainer.main_eval \
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

# MATH Eval
if [ "${EVAL_MATH}" = true ]; then
    $PYBIN -m verl.trainer.main_eval \
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

# OPENMATHINSTRUCT2 Eval
if [ "${OPENMATHINSTRUCT2}" = true ] && [ -n "${OPENMATHINSTRUCT2_OUTPUT}" ]; then
    $PYBIN -m verl.trainer.main_eval \
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

# Majority Voting
if [ "${N_SAMPLES}" -gt 1 ]; then
    if [ "${EVAL_GSM8K}" = true ]; then
        $PYBIN -m scripts.evaluate_majority_voting \
            --input_file "${GSM8K_OUTPUT}" \
            --dataset_type "gsm8k" | tee "${OUTPUT_DIR}/gsm8k_majority_results.txt"
    fi
    if [ "${EVAL_MATH}" = true ]; then
        $PYBIN -m scripts.evaluate_majority_voting \
            --input_file "${MATH_OUTPUT}" \
            --dataset_type "math" | tee "${OUTPUT_DIR}/math_majority_results.txt"
    fi
fi
