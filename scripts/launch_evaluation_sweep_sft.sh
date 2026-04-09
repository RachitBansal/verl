#!/bin/bash
#
# Launcher script for OLMo-2 SFT evaluation sweep
# Discovers SFT experiment directories and evaluates the latest -hf checkpoint
# in each one using scripts/evaluate_olmo2_math.sh.
#
# Usage:
#   bash scripts/launch_evaluation_sweep_sft.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

#############################################
# CONFIGURATION
#############################################

# Base SFT experiment directory (each subdir like OLMo2-1B-step10000-stage2-openmathgsm8k)
CHECKPOINT_BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Base directory for verl
BASE_DIR="/n/home05/sqin/rl_pretrain/verl/"
EVAL_SCRIPT="${BASE_DIR}/scripts/evaluate_olmo2_math_rl.sh"

# N_SAMPLES values to test (top-k generations per prompt)
N_SAMPLES_LIST=(32)

# SLURM Configuration
SLURM_PARTITION="kempner"
SLURM_ACCOUNT="kempner_dam_lab"
SLURM_TIME="4:00:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=1
SLURM_CPUS_PER_TASK=24
SLURM_MEM="200GB"

# Output directories
SBATCH_DIR="${BASE_DIR}/sbatch_jobs_sft"
LOGS_DIR="${BASE_DIR}/logs"
mkdir -p "${SBATCH_DIR}"
mkdir -p "${LOGS_DIR}"

#############################################
# DISCOVER SFT EXPERIMENTS AND LATEST -hf CKPTS
#############################################

echo "================================================"
echo "OLMo-2 SFT Evaluation Sweep Launcher"
echo "================================================"
echo "Experiment base directory: ${CHECKPOINT_BASE_DIR}"
echo ""

CHECKPOINT_PATHS=()
CHECKPOINT_NAMES=()
CHECKPOINT_STEPS=()

echo "Discovering SFT checkpoints..."
while IFS= read -r -d '' checkpoint; do
    experiment_name=$(basename "$(dirname "${checkpoint}")")
    checkpoint_step_dir=$(basename "${checkpoint}")
    checkpoint_step=${checkpoint_step_dir%-hf}
    checkpoint_step=${checkpoint_step#step}
    CHECKPOINT_PATHS+=("${checkpoint}")
    CHECKPOINT_NAMES+=("${experiment_name}")
    CHECKPOINT_STEPS+=("${checkpoint_step}")
    echo "  Found: ${experiment_name} (${checkpoint_step_dir}) -> ${checkpoint}"
done < <(find "${CHECKPOINT_BASE_DIR}" -maxdepth 2 -type d -name "step*-hf" -path "*/OLMo2-1B-step1000-stage2*/*" -print0 | sort -z)

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "ERROR: No SFT experiment checkpoints found under ${CHECKPOINT_BASE_DIR}"
    exit 1
fi

echo ""
echo "Experiments with checkpoints: ${#CHECKPOINT_PATHS[@]}"
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

    model_name="${experiment}-step${checkpoint_step}-sft"
    model_path="${checkpoint}"

    for n_samples in "${N_SAMPLES_LIST[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))

        job_name="sft-eval-${experiment}-step${checkpoint_step}-${n_samples}samples"
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
echo "Experiment: ${experiment}"
echo "Model Path: ${model_path}"
echo "N_SAMPLES: ${n_samples}"
echo "Started at: \$(date)"
echo "================================================"
echo ""

cd "${BASE_DIR}"

# Run SFT evaluation (GSM8K enabled, MATH disabled by default)
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
        checkpoint_step="${CHECKPOINT_STEPS[$idx]}"

        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="sft-eval-${experiment}-step${checkpoint_step}-${n_samples}samples"
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
    for idx in "${!CHECKPOINT_PATHS[@]}"; do
        experiment="${CHECKPOINT_NAMES[$idx]}"
        checkpoint_step="${CHECKPOINT_STEPS[$idx]}"
        for n_samples in "${N_SAMPLES_LIST[@]}"; do
            job_name="sft-eval-${experiment}-step${checkpoint_step}-${n_samples}samples"
            echo "  sbatch ${SBATCH_DIR}/${job_name}.sbatch"
        done
    done
fi



