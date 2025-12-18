#!/bin/bash
#
# Launcher script for OLMo-2 RL evaluation sweep
# Discovers RL checkpoints and creates sbatch jobs to run
# scripts/evaluate_olmo2_math_rl.sh on each checkpoint.
#
# Usage:
#   bash scripts/launch_evaluation_sweep_rl.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

#############################################
# CONFIGURATION
#############################################

# Base checkpoint directory (expects .../experiments/<exp>/hf_model/step800)
CHECKPOINT_BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Base directory for verl
BASE_DIR="/n/netscratch/dam_lab/Lab/brachit/rl/verl"
EVAL_SCRIPT="${BASE_DIR}/scripts/evaluate_olmo2_math_rl.sh"

# N_SAMPLES values to test (used as top-k generations per prompt)
N_SAMPLES_LIST=(1 8 32 128)

# SLURM Configuration
SLURM_PARTITION="kempner_h100"
SLURM_ACCOUNT="kempner_dam_lab"
SLURM_TIME="24:00:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=1
SLURM_CPUS_PER_TASK=24
SLURM_MEM="500GB"

# Output directories
SBATCH_DIR="${BASE_DIR}/sbatch_jobs_rl"
LOGS_DIR="${BASE_DIR}/logs"
mkdir -p "${SBATCH_DIR}"
mkdir -p "${LOGS_DIR}"

#############################################
# DISCOVER CHECKPOINTS
#############################################

echo "================================================"
echo "OLMo-2 RL Evaluation Sweep Launcher"
echo "================================================"
echo "Checkpoint directory: ${CHECKPOINT_BASE_DIR}"
echo ""

CHECKPOINT_PATHS=()
CHECKPOINT_NAMES=()

echo "Discovering checkpoints..."
while IFS= read -r -d '' checkpoint; do
    # experiments/<exp>/hf_model/step800 -> extract <exp>
    experiment_name=$(basename "$(dirname "$(dirname "${checkpoint}")")")
    CHECKPOINT_PATHS+=("${checkpoint}")
    CHECKPOINT_NAMES+=("${experiment_name}")
    echo "  Found: ${experiment_name} -> ${checkpoint}"
done < <(find "${CHECKPOINT_BASE_DIR}" -maxdepth 3 -type d -path "${CHECKPOINT_BASE_DIR}/*/hf_model/step800" -print0 | sort -z)

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found at ${CHECKPOINT_BASE_DIR}/*/hf_model/step800"
    exit 1
fi

echo ""
echo "Found ${#CHECKPOINT_PATHS[@]} checkpoints"
echo "N_SAMPLES values: ${N_SAMPLES_LIST[@]}"
echo "Total combinations: $((${#CHECKPOINT_PATHS[@]} * ${#N_SAMPLES_LIST[@]}))"
echo ""

#############################################
# GENERATE SBATCH SCRIPTS
#############################################

echo "Generating sbatch scripts..."

JOB_IDS=()
JOB_COUNT=0

for idx in "${!CHECKPOINT_PATHS[@]}"; do
    checkpoint="${CHECKPOINT_PATHS[$idx]}"
    experiment="${CHECKPOINT_NAMES[$idx]}"

    model_name="${experiment}-step800-rl"
    model_path="${checkpoint}"

    for n_samples in "${N_SAMPLES_LIST[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))

        job_name="rl-eval-${experiment}-step800-${n_samples}samples"
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

set -e
set -u

echo "================================================"
echo "Job: ${job_name}"
echo "Model Path: ${model_path}"
echo "N_SAMPLES: ${n_samples}"
echo "Started at: \$(date)"
echo "================================================"
echo ""

cd "${BASE_DIR}"

# Run RL evaluation with explicit positional arguments expected by the script
bash "${EVAL_SCRIPT}" "${model_path}" "${model_name}" "${n_samples}"

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

    for idx in "${!CHECKPOINT_PATHS[@]}"; do
        experiment="${CHECKPOINT_NAMES[$idx]}"

        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="rl-eval-${experiment}-step800-${n_samples}samples"
            sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"

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
    for idx in "${!CHECKPOINT_PATHS[@]}"; do
        experiment="${CHECKPOINT_NAMES[$idx]}"
        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="rl-eval-${experiment}-step800-${n_samples}samples"
            echo "  sbatch ${SBATCH_DIR}/${job_name}.sbatch"
        done
    done
fi


