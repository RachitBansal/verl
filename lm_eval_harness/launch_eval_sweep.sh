#!/bin/bash
#
# Launcher script for lm_eval capability evaluation sweep.
# Discovers HF checkpoints and submits sbatch jobs to evaluate each.
#
# Usage:
#   bash lm_eval_harness/launch_eval_sweep.sh
#
# Env-var overrides:
#   TASK_CONFIG=eval_tasks_nonmath.yaml    # task group YAML (default)
#   LIMIT=100                              # per-task sample limit for screening (unset = full)
#   EXP_PATTERN="..."                      # override checkpoint glob
#
# Screening example (100 samples, non-math set):
#   LIMIT=100 bash lm_eval_harness/launch_eval_sweep.sh
#

set -e
set -u

#############################################
# CONFIGURATION
#############################################

CHECKPOINT_BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

# Experiment pattern to match (glob)
EXP_PATTERN="${EXP_PATTERN:-OLMo2-1B_step*_interleave_twoloader_n32_sft_50000_ppo_0_rgsm}"

# Task config + per-task limit (passed through to run_eval.py)
TASK_CONFIG="${TASK_CONFIG:-eval_tasks_nonmath.yaml}"
LIMIT="${LIMIT:-}"

BASE_DIR="/n/home05/sqin/rl_pretrain/verl"

# SLURM Configuration
SLURM_PARTITION="kempner"
SLURM_ACCOUNT="kempner_dam_lab"
SLURM_TIME="00:30:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=1
SLURM_CPUS_PER_TASK=16
SLURM_MEM="128GB"

# Output directories
SBATCH_DIR="${BASE_DIR}/sbatch_jobs_lm_eval"
LOGS_DIR="${BASE_DIR}/logs"
mkdir -p "${SBATCH_DIR}"
mkdir -p "${LOGS_DIR}"

#############################################
# DISCOVER CHECKPOINTS (last step per experiment)
#############################################

echo "================================================"
echo "LM Eval Capability Sweep Launcher"
echo "================================================"
echo "Checkpoint directory: ${CHECKPOINT_BASE_DIR}"
echo "Experiment pattern:   ${EXP_PATTERN}"
echo "Task config:          ${TASK_CONFIG}"
echo "Per-task limit:       ${LIMIT:-<full>}"
echo ""

CHECKPOINT_PATHS=()
CHECKPOINT_NAMES=()

echo "Discovering checkpoints (last step per experiment)..."
for exp_dir in "${CHECKPOINT_BASE_DIR}"/${EXP_PATTERN}; do
    [ -d "${exp_dir}/hf_model" ] || continue
    # Find the last (highest) step directory
    last_step=$(ls "${exp_dir}/hf_model/" | sort -t'p' -k2 -n | tail -1)
    [ -z "${last_step}" ] && continue
    checkpoint="${exp_dir}/hf_model/${last_step}"
    experiment_name=$(basename "${exp_dir}")
    CHECKPOINT_PATHS+=("${checkpoint}")
    CHECKPOINT_NAMES+=("${experiment_name}_${last_step}")
    echo "  ${experiment_name} -> ${last_step}"
done

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found"
    exit 1
fi

echo ""
echo "Found ${#CHECKPOINT_PATHS[@]} checkpoints"
echo ""

#############################################
# GENERATE SBATCH SCRIPTS
#############################################

echo "Generating sbatch scripts..."

JOB_COUNT=0

for idx in "${!CHECKPOINT_PATHS[@]}"; do
    checkpoint="${CHECKPOINT_PATHS[$idx]}"
    name="${CHECKPOINT_NAMES[$idx]}"

    job_name="lm-eval-${name}"
    sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"
    JOB_COUNT=$((JOB_COUNT + 1))

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

export PATH="/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf/bin:\$PATH"

echo "================================================"
echo "LM Eval: ${name}"
echo "Model: ${checkpoint}"
echo "Started at: \$(date)"
echo "================================================"

cd "${BASE_DIR}"

python lm_eval_harness/run_eval.py --model_path "${checkpoint}" --task_config "lm_eval_harness/${TASK_CONFIG}"${LIMIT:+ --limit ${LIMIT}}

echo ""
echo "Completed at: \$(date)"
EOF

    chmod +x "${sbatch_file}"
    echo "  Created: ${sbatch_file}"
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
    JOB_IDS=()

    for idx in "${!CHECKPOINT_PATHS[@]}"; do
        name="${CHECKPOINT_NAMES[$idx]}"
        job_name="lm-eval-${name}"
        sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"

        job_id=$(sbatch "${sbatch_file}" | grep -oP '\d+')
        JOB_IDS+=("${job_id}")
        echo "  Submitted: ${job_name} (Job ID: ${job_id})"
    done

    echo ""
    echo "================================================"
    echo "All ${#JOB_IDS[@]} jobs submitted!"
    echo "Job IDs: ${JOB_IDS[*]}"
    echo "Monitor: squeue -u \$USER"
    echo "Logs: ${LOGS_DIR}"
    echo "================================================"
else
    echo ""
    echo "Jobs not submitted. To submit manually:"
    for idx in "${!CHECKPOINT_PATHS[@]}"; do
        name="${CHECKPOINT_NAMES[$idx]}"
        echo "  sbatch ${SBATCH_DIR}/lm-eval-${name}.sbatch"
    done
fi
