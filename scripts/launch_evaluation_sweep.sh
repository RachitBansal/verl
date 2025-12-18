#!/bin/bash
#
# Launcher script for OLMo-2 evaluation sweep
# Discovers all intermediate checkpoints and creates sbatch jobs for evaluation
# with different N_SAMPLES values for both MATH and GSM-8K datasets
#
# Usage:
#   bash scripts/launch_evaluation_sweep.sh
#
# This script will:
#   1. Discover all -hf checkpoints in the checkpoint directory
#   2. Generate sbatch scripts for each checkpoint and N_SAMPLES combination
#   3. Submit all jobs to SLURM
#

set -e  # Exit on error
set -u  # Exit on undefined variable

#############################################
# CONFIGURATION
#############################################

# Base checkpoint directory
CHECKPOINT_BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B"

# Base directory for verl
BASE_DIR="/n/netscratch/dam_lab/Lab/brachit/rl/verl"
EVAL_SCRIPT="${BASE_DIR}/scripts/evaluate_olmo2_math.sh"

# N_SAMPLES values to test
N_SAMPLES_LIST=(128)

# SLURM Configuration
SLURM_PARTITION="kempner"
SLURM_ACCOUNT="kempner_dam_lab"
SLURM_TIME="24:00:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=1
SLURM_CPUS_PER_TASK=24
SLURM_MEM="200GB"

# Output directory for generated sbatch scripts
SBATCH_DIR="${BASE_DIR}/sbatch_jobs"
mkdir -p "${SBATCH_DIR}"

# Logs directory
LOGS_DIR="${BASE_DIR}/logs"
mkdir -p "${LOGS_DIR}"

#############################################
# DISCOVER CHECKPOINTS
#############################################

echo "================================================"
echo "OLMo-2 Evaluation Sweep Launcher"
echo "================================================"
echo "Checkpoint directory: ${CHECKPOINT_BASE_DIR}"
echo ""

# Find all -hf checkpoints
echo "Discovering checkpoints..."
CHECKPOINTS=()
while IFS= read -r -d '' checkpoint; do
    checkpoint_name=$(basename "${checkpoint}")
    CHECKPOINTS+=("${checkpoint_name}")
    echo "  Found: ${checkpoint_name}"
done < <(find "${CHECKPOINT_BASE_DIR}" -maxdepth 1 -type d -name "*-hf" -print0 | sort -z)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No -hf checkpoints found in ${CHECKPOINT_BASE_DIR}"
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
    # Extract step number from checkpoint name (e.g., "step22000-hf" -> "step22000")
    step_name="${checkpoint%-hf}"
    model_name="1B-${step_name}"
    model_path="${CHECKPOINT_BASE_DIR}/${checkpoint}"
    
    for n_samples in "${N_SAMPLES_LIST[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))
        
        # Create unique job name
        job_name="eval-${step_name}-${n_samples}samples"
        sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"
        
        # Generate sbatch script
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

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "================================================"
echo "Job: ${job_name}"
echo "Checkpoint: ${checkpoint}"
echo "Model Path: ${model_path}"
echo "N_SAMPLES: ${n_samples}"
echo "Started at: \$(date)"
echo "================================================"
echo ""

# Change to base directory
cd "${BASE_DIR}"

# Run evaluation for both GSM-8K and MATH
bash "${EVAL_SCRIPT}" "${model_path}" "${model_name}" "${n_samples}" "true" "false"

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
            job_name="eval-${step_name}-${n_samples}samples"
            sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"
            
            # Submit job and capture job ID
            job_id=$(sbatch "${sbatch_file}" | grep -oP '\d+')
            JOB_IDS+=("${job_id}")
            
            echo "  Submitted: ${job_name} (Job ID: ${job_id})"
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
            job_name="eval-${step_name}-${n_samples}samples"
            echo "  sbatch ${SBATCH_DIR}/${job_name}.sbatch"
        done
    done
fi






