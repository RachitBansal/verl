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

# Manually specify experiment checkpoint directories (each should contain hf_model/step* subdirs)
# Format: path/to/experiment (the script will look for hf_model/step* inside each)
EXPERIMENT_DIRS=(
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n5_easy_epochs5_57501168"
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n5_balanced_epochs5_57501214"
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n5_hard_epochs5_57501117"
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n64_balanced_epochs5_57501202"
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n64_easy_epochs5_57501190"
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n64_hard_epochs5_57501197"
    "/n/netscratch/dam_lab/Everyone/rl_rollouts/experiments/fixed_n64_balanced_epochs5_57501202"
)

# Base directory for verl
BASE_DIR="/n/netscratch/dam_lab/Lab/brachit/rollouts/verl"
EVAL_SCRIPT="${BASE_DIR}/scripts/evaluate_olmo2_math_rl.sh"

# N_SAMPLES values to test (used as top-k generations per prompt)
N_SAMPLES_LIST=(8)

# SLURM Configuration
SLURM_PARTITION="kempner"
SLURM_ACCOUNT="kempner_dam_lab"
SLURM_TIME="4:00:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=1
SLURM_CPUS_PER_TASK=24
SLURM_MEM="200GB"

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
echo "Experiment directories to process: ${#EXPERIMENT_DIRS[@]}"
echo ""

CHECKPOINT_PATHS=()
CHECKPOINT_NAMES=()
CHECKPOINT_STEPS=()

echo "Discovering checkpoints in specified directories..."
for exp_dir in "${EXPERIMENT_DIRS[@]}"; do
    experiment_name=$(basename "${exp_dir}")
    hf_model_dir="${exp_dir}/hf_model"

    if [ ! -d "${hf_model_dir}" ]; then
        echo "  WARNING: No hf_model directory found in ${exp_dir}, skipping..."
        continue
    fi

    # Find all step* directories within hf_model/
    while IFS= read -r -d '' checkpoint; do
        checkpoint_step_dir=$(basename "${checkpoint}")
        checkpoint_step=${checkpoint_step_dir#step}
        CHECKPOINT_PATHS+=("${checkpoint}")
        CHECKPOINT_NAMES+=("${experiment_name}")
        CHECKPOINT_STEPS+=("${checkpoint_step}")
        echo "  Found: ${experiment_name} (${checkpoint_step_dir}) -> ${checkpoint}"
    done < <(find "${hf_model_dir}" -maxdepth 1 -type d -name "step*" -print0 | sort -z)
done

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in any of the specified experiment directories"
    echo "Expected structure: <experiment_dir>/hf_model/step*"
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
    checkpoint_step="${CHECKPOINT_STEPS[$idx]}"

    model_name="${experiment}-step${checkpoint_step}-rl"
    model_path="${checkpoint}"

    for n_samples in "${N_SAMPLES_LIST[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))

        job_name="rl-eval-${experiment}-step${checkpoint_step}-${n_samples}samples"
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
bash "${EVAL_SCRIPT}" "${model_path}" "${model_name}" "${n_samples}" true false

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
            checkpoint_step="${CHECKPOINT_STEPS[$idx]}"
            job_name="rl-eval-${experiment}-step${checkpoint_step}-${n_samples}samples"
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
        checkpoint_step="${CHECKPOINT_STEPS[$idx]}"
        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="rl-eval-${experiment}-step${checkpoint_step}-${n_samples}samples"
            echo "  sbatch ${SBATCH_DIR}/${job_name}.sbatch"
        done
    done
fi


