#!/bin/bash
# Submit jobs for resumption (crashed) or extension (completed)
#
# Usage:
#   ./submit_all_resume_jobs.sh                    # Submit all crashed jobs
#   ./submit_all_resume_jobs.sh n5                 # Submit only n=5 crashed jobs
#   ./submit_all_resume_jobs.sh n64                # Submit only n=64 crashed jobs
#   ./submit_all_resume_jobs.sh extend             # Submit only jobs to extend
#   ./submit_all_resume_jobs.sh extend n64         # Submit only n=64 jobs to extend
#   ./submit_all_resume_jobs.sh all                # Submit both crashed and extend jobs
#
# To customize which jobs to resume/extend, edit the CRASHED_JOBS and EXTEND_JOBS arrays below.

cd "$(dirname "$0")"

# ============================================================================
# CRASHED JOBS - Jobs that didn't finish and need to be resumed
# Format: ["job_name"]="info about current progress"
# ============================================================================
declare -A CRASHED_JOBS=(
    # ["fixed_n5_easy_epochs100_58022463"]="step 280/1560"
    # ["fixed_n5_hard_epochs100_58022458"]="step 400/1560"
    # ["fixed_n5_balanced_epochs100_58022453"]="step 300/1560"
    # ["fixed_n5_hard_epochs5_57501117"]="step 1200/1560"
    # ["fixed_n5_easy_epochs5_57501168"]="step 1200/1560"
    # ["fixed_n5_balanced_epochs5_57501214"]="step 1200/1560"
    # ["fixed_n5_hard_epochs10_57675814"]="step 750/1560"
    # ["fixed_n5_easy_epochs10_57675780"]="step 1240/1560"
)

# ============================================================================
# EXTEND JOBS - Completed jobs to continue training for more epochs
# Format: ["job_name"]="new_total_epochs"
# Example: A job that completed 10 epochs, extend to 20 epochs
# ============================================================================
declare -A EXTEND_JOBS=(
    ["fixed_n64_hard_epochs10_57675738"]="20"
    # ["fixed_n64_easy_epochs10_57675685"]="20"
    # ["fixed_n5_hard_epochs10_57675814"]="20"
    # ["fixed_n5_easy_epochs10_57675780"]="20"
)

# Parse arguments
MODE="extend"  # Default: only crashed jobs
N_FILTER="all"  # Default: all n values

for arg in "$@"; do
    case "$arg" in
        extend)
            MODE="extend"
            ;;
        all)
            MODE="all"
            ;;
        n5|n64)
            N_FILTER="$arg"
            ;;
    esac
done

echo "=============================================="
echo "Resume/Extend Training Jobs"
echo "Mode: $MODE | N Filter: $N_FILTER"
echo "=============================================="
echo ""

submitted=0

# Helper function to check n filter
check_n_filter() {
    local job=$1
    if [ "$N_FILTER" = "n5" ] && [[ ! "$job" =~ "n5" ]]; then
        return 1
    fi
    if [ "$N_FILTER" = "n64" ] && [[ ! "$job" =~ "n64" ]]; then
        return 1
    fi
    return 0
}

# Submit crashed jobs (resume to original epoch target)
if [ "$MODE" = "crashed" ] || [ "$MODE" = "all" ]; then
    echo "--- CRASHED JOBS (resuming to original target) ---"
    for job in "${!CRASHED_JOBS[@]}"; do
        info="${CRASHED_JOBS[$job]}"
        
        # Check n filter
        if ! check_n_filter "$job"; then
            continue
        fi
        
        # Skip jobs marked as DONE
        if [[ "$info" =~ "DONE" ]]; then
            echo "SKIP: $job ($info)"
            continue
        fi
        
        echo "Submitting: $job ($info)"
        RESUME_JOB=$job sbatch resume_crashed_jobs.sbatch
        ((submitted++))
        
        sleep 2
    done
    echo ""
fi

# Submit extend jobs (continue completed jobs for more epochs)
if [ "$MODE" = "extend" ] || [ "$MODE" = "all" ]; then
    echo "--- EXTEND JOBS (continuing for more epochs) ---"
    for job in "${!EXTEND_JOBS[@]}"; do
        new_epochs="${EXTEND_JOBS[$job]}"
        
        # Check n filter
        if ! check_n_filter "$job"; then
            continue
        fi
        
        echo "Submitting: $job (extending to $new_epochs epochs)"
        RESUME_JOB=$job NEW_EPOCHS=$new_epochs sbatch resume_crashed_jobs.sbatch
        ((submitted++))
        
        sleep 2
    done
    echo ""
fi

echo "=============================================="
echo "Submitted $submitted job(s). Check status with: squeue -u \$USER"
echo "=============================================="
