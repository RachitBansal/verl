#!/bin/bash
#
# Launcher script for OLMo-2 4B base model evaluation sweep
# Discovers step*-hf checkpoints under OLMo2-4B-stage1-50B and creates sbatch
# jobs that run scripts/evaluate_olmo2_math_4b.sh on each.
#
# Usage:
#   bash scripts/launch_evaluation_sweep_4b.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

#############################################
# CONFIGURATION
#############################################

# Base checkpoint directory
CHECKPOINT_BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-4B-stage1-50B"
MODEL_NAME="4B-stage1-50B"

# Specific checkpoint steps to evaluate (filters discovery to just these)
CHECKPOINT_STEPS=(5000 14000)

# Base directory for verl
BASE_DIR="/n/home05/sqin/rl_pretrain/verl/"
EVAL_SCRIPT="${BASE_DIR}/scripts/evaluate_olmo2_math_4b.sh"

# N_SAMPLES values to test (32 for pass@32 / majority voting)
N_SAMPLES_LIST=(32)

# SLURM Configuration
SLURM_PARTITION="kempner_h100"
SLURM_ACCOUNT="kempner_barak_lab"
SLURM_TIME="20:00:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=2
SLURM_CPUS_PER_TASK=24
SLURM_MEM="250GB"

# Evaluation flags (baked into each sbatch job at submission time)
EVAL_GSM8K=true
EVAL_MATH=true

# Output directories
SBATCH_DIR="${BASE_DIR}/sbatch_jobs_4b"
LOGS_DIR="${BASE_DIR}/logs"
mkdir -p "${SBATCH_DIR}"
mkdir -p "${LOGS_DIR}"

#############################################
# DISCOVER CHECKPOINTS
#############################################

echo "================================================"
echo "OLMo-2 4B Base Evaluation Sweep Launcher"
echo "================================================"
echo "Checkpoint directory: ${CHECKPOINT_BASE_DIR}"
echo "Steps requested: ${CHECKPOINT_STEPS[@]}"
echo ""

CHECKPOINTS=()
echo "Discovering checkpoints..."
for step in "${CHECKPOINT_STEPS[@]}"; do
    ckpt_name="step${step}-hf"
    ckpt_path="${CHECKPOINT_BASE_DIR}/${ckpt_name}"
    if [ ! -d "${ckpt_path}" ]; then
        echo "ERROR: Checkpoint not found: ${ckpt_path}"
        exit 1
    fi
    CHECKPOINTS+=("${ckpt_name}")
    echo "  Found: ${ckpt_name}"
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints discovered."
    exit 1
fi

echo ""
echo "Found ${#CHECKPOINTS[@]} checkpoints"
echo "N_SAMPLES values: ${N_SAMPLES_LIST[@]}"
echo "Total combinations: $((${#CHECKPOINTS[@]} * ${#N_SAMPLES_LIST[@]}))"
echo ""

#############################################
# GENERATE SBATCH SCRIPTS
#############################################

echo "Generating sbatch scripts..."

JOB_IDS=()
JOB_COUNT=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    step_name="${checkpoint%-hf}"
    model_name="${MODEL_NAME}-${step_name}"
    model_path="${CHECKPOINT_BASE_DIR}/${checkpoint}"

    for n_samples in "${N_SAMPLES_LIST[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))

        job_name="eval-4b-${step_name}-${n_samples}samples"
        sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"

        cat > "${sbatch_file}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/${job_name}.out
#SBATCH --error=${LOGS_DIR}/${job_name}.err
#SBATCH --time=${SLURM_TIME}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${SLURM_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --exclude=holygpu8a19102

set -e
set -u

echo "================================================"
echo "Job: ${job_name}"
echo "Checkpoint: ${checkpoint}"
echo "Model Path: ${model_path}"
echo "N_SAMPLES: ${n_samples}"
echo "Started at: \$(date)"
echo "================================================"
echo ""

cd "${BASE_DIR}"

# Run evaluation with explicit positional arguments expected by the script
bash "${EVAL_SCRIPT}" "${model_path}" "${model_name}" "${n_samples}" "${EVAL_GSM8K}" "${EVAL_MATH}"

echo ""
echo "================================================"
echo "Job completed at: \$(date)"
echo "================================================"
EOF

        chmod +x "${sbatch_file}"
        echo "  Created: ${sbatch_file}"
    done
done

echo ""
echo "Generated ${JOB_COUNT} sbatch scripts"
echo ""

#############################################
# SUBMIT JOBS
#############################################

read -p "Submit all ${JOB_COUNT} jobs to SLURM? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting jobs..."

    for checkpoint in "${CHECKPOINTS[@]}"; do
        step_name="${checkpoint%-hf}"

        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="eval-4b-${step_name}-${n_samples}samples"
            sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"

            job_id=$(sbatch "${sbatch_file}" | grep -oP '\d+')
            JOB_IDS+=("${job_id}")

            echo "  Submitted: ${job_name} (Job ID: ${job_id})"
            sleep 2
        done
    done

    echo ""
    echo "================================================"
    echo "All jobs submitted!"
    echo "================================================"
    echo "Total jobs: ${#JOB_IDS[@]}"
    echo ""
    echo "Job IDs: ${JOB_IDS[@]}"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Cancel all jobs with:"
    echo "  scancel ${JOB_IDS[@]}"
    echo ""
    echo "View logs in: ${LOGS_DIR}"
    echo "================================================"
else
    echo ""
    echo "Jobs not submitted. To submit manually, run:"
    echo "  sbatch ${SBATCH_DIR}/*.sbatch"
    echo ""
    echo "Or submit individually:"
    for checkpoint in "${CHECKPOINTS[@]}"; do
        step_name="${checkpoint%-hf}"
        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="eval-4b-${step_name}-${n_samples}samples"
            echo "  sbatch ${SBATCH_DIR}/${job_name}.sbatch"
        done
    done
fi
