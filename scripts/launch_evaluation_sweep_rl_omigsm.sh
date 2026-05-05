#!/bin/bash
#
# Launcher: OMI-GSM evaluation sweep over Direct-RL OLMo-2 1B checkpoints.
# For each (experiment, last_step) pair below, generates an sbatch job that runs
# scripts/evaluate_olmo2_math_rl.sh with EVAL_OMI_GSM=true (and all others false),
# producing 0-shot N=32 omi_gsm_predictions + omi_gsm_results + majority pass@k.
#
# Usage:
#   bash scripts/launch_evaluation_sweep_rl_omigsm.sh

set -e
set -u

#############################################
# CONFIGURATION
#############################################

CHECKPOINT_BASE_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"
BASE_DIR="/n/home05/sqin/rl_pretrain/verl/"
EVAL_SCRIPT="${BASE_DIR}/scripts/evaluate_olmo2_math_rl.sh"

N_SAMPLES=32

# SLURM
SLURM_PARTITION="kempner"
SLURM_ACCOUNT="kempner_barak_lab"
SLURM_TIME="20:00:00"
SLURM_NODES=1
SLURM_GPUS_PER_NODE=1
SLURM_CPUS_PER_TASK=24
SLURM_MEM="250GB"

# Eval flags: gsm8k math omi_gsm omi_math
EVAL_GSM8K=false
EVAL_MATH=false
EVAL_OMI_GSM=true
EVAL_OMI_MATH=false

SBATCH_DIR="${BASE_DIR}/sbatch_jobs_rl"
LOGS_DIR="${BASE_DIR}/logs"
mkdir -p "${SBATCH_DIR}"
mkdir -p "${LOGS_DIR}"

#############################################
# HARDCODED CHECKPOINT LIST
# Format: "experiment_name:last_step"
#############################################

CKPTS=(
    "olmo2_1b_step1000_omigsm8k_n32:400"
    "olmo2_1b_step2000_omigsm8k_n32:1560"
    "olmo2_1b_step2000_omigsm8k_n32_v2:1000"
    "olmo2_1b_step2000_omigsm8k_n32_v3:1000"
    "olmo2_1b_step3000_omigsm8k_n32:1560"
    "olmo2_1b_step3000_omigsm8k_n32_v2:1000"
    "olmo2_1b_step3000_omigsm8k_n32_v3:1200"
    "olmo2_1b_step3000_omigsm8k_n32_v4:600"
    "olmo2_1b_step5000_omigsm8k_n32:1560"
    "olmo2_1b_step5000_omigsm8k_n32_v2:1000"
    "olmo2_1b_step5000_omigsm8k_n32_v3:1400"
    "olmo2_1b_step6000_omigsm8k_n32:1400"
    "olmo2_1b_step7000_omigsm8k_n32:1000"
    "olmo2_1b_step10000_omigsm8k_n32:1560"
    "olmo2_1b_step14000_omigsm8k_n32:1200"
    "olmo2_1b_step22000_omigsm8k_n32:1000"
)

#############################################
# GENERATE SBATCH SCRIPTS
#############################################

echo "================================================"
echo "OMI-GSM RL Evaluation Sweep Launcher"
echo "================================================"
echo "Checkpoints: ${#CKPTS[@]}"
echo "N_SAMPLES: ${N_SAMPLES}"
echo ""

JOB_NAMES=()

for entry in "${CKPTS[@]}"; do
    experiment="${entry%%:*}"
    checkpoint_step="${entry##*:}"
    model_path="${CHECKPOINT_BASE_DIR}/${experiment}/hf_model/step${checkpoint_step}"
    model_name="${experiment}-step${checkpoint_step}-rl"

    if [ ! -f "${model_path}/config.json" ]; then
        echo "WARN: missing config.json at ${model_path} — skipping"
        continue
    fi

    job_name="omigsm-eval-${experiment}-step${checkpoint_step}-${N_SAMPLES}samples"
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
echo "Model Path: ${model_path}"
echo "N_SAMPLES: ${N_SAMPLES}"
echo "Started at: \$(date)"
echo "================================================"
echo ""

cd "${BASE_DIR}"

bash "${EVAL_SCRIPT}" "${model_path}" "${model_name}" "${N_SAMPLES}" "${EVAL_GSM8K}" "${EVAL_MATH}" "${EVAL_OMI_GSM}" "${EVAL_OMI_MATH}"

echo ""
echo "================================================"
echo "Job completed at: \$(date)"
echo "================================================"
EOF

    chmod +x "${sbatch_file}"
    JOB_NAMES+=("${job_name}")
    echo "  Created: ${sbatch_file}"
done

echo ""
echo "Generated ${#JOB_NAMES[@]} sbatch scripts"
echo ""

#############################################
# SUBMIT JOBS
#############################################

read -p "Submit all ${#JOB_NAMES[@]} jobs to SLURM? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    JOB_IDS=()
    for job_name in "${JOB_NAMES[@]}"; do
        sbatch_file="${SBATCH_DIR}/${job_name}.sbatch"
        job_id=$(sbatch "${sbatch_file}" | grep -oP '\d+')
        JOB_IDS+=("${job_id}")
        echo "  Submitted: ${job_name} (Job ID: ${job_id})"
        sleep 2
    done

    echo ""
    echo "================================================"
    echo "All jobs submitted!"
    echo "================================================"
    echo "Total: ${#JOB_IDS[@]}"
    echo "Job IDs: ${JOB_IDS[@]}"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Cancel:  scancel ${JOB_IDS[@]}"
else
    echo "Jobs not submitted. Submit manually with:"
    echo "  for f in ${SBATCH_DIR}/omigsm-eval-*.sbatch; do sbatch \"\$f\"; done"
fi
