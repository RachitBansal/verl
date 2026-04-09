#!/bin/bash
# Script to prune checkpoints in the experiments directory.
# - For RL runs, keep only checkpoints where step % 500 == 0
# - For SFT runs, keep only checkpoints where step % 1000 == 0
# - Always keep the latest checkpoint
# - For kept non-latest checkpoints, delete optimizer state files under actor/

set -u

EXPERIMENTS_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments/"

# Set to 1 for dry-run (just print what would be deleted), 0 to actually delete
DRY_RUN=${DRY_RUN:-0}

total_checkpoint_dirs_to_delete=0
total_optim_files_to_delete=0

detect_run_type() {
    local experiment_name="$1"
    local lower_name
    lower_name=$(echo "$experiment_name" | tr '[:upper:]' '[:lower:]')

    # Mixed runs like sft_rl should use the RL retention rule.
    if [[ "$lower_name" == *"rl"* ]]; then
        echo "rl"
        return
    fi

    if [[ "$lower_name" == *"sft"* ]] || [[ "$lower_name" == *"sfted"* ]] || [[ "$lower_name" == *"stage2"* ]]; then
        echo "sft"
        return
    fi

    # Default to RL because it is the less aggressive retention rule.
    echo "rl"
}

retention_interval_for_run_type() {
    local run_type="$1"

    case "$run_type" in
        rl)
            echo 500
            ;;
        sft)
            echo 1000
            ;;
        *)
            echo 500
            ;;
    esac
}

echo "============================================"
echo "Checkpoint Cleanup Script"
echo "============================================"
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Dry run: $DRY_RUN (set DRY_RUN=0 to actually delete)"
echo "RL retention: keep steps where step % 500 == 0"
echo "SFT retention: keep steps where step % 1000 == 0"
echo "Latest checkpoint is always preserved"
echo "============================================"
echo ""

shopt -s nullglob

# Iterate over all experiment directories
for experiment_dir in "$EXPERIMENTS_DIR"/*/; do
    experiment_name=$(basename "$experiment_dir")

    mapfile -t checkpoints < <(
        find "$experiment_dir" -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | sort -V
    )

    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "[$experiment_name] No checkpoints found, skipping."
        continue
    fi

    run_type=$(detect_run_type "$experiment_name")
    retention_interval=$(retention_interval_for_run_type "$run_type")

    last_checkpoint="${checkpoints[${#checkpoints[@]}-1]}"
    last_checkpoint_name=$(basename "$last_checkpoint")
    last_step=${last_checkpoint_name#global_step_}

    echo "[$experiment_name] Run type: $run_type, retention interval: $retention_interval"
    echo "[$experiment_name] Found ${#checkpoints[@]} checkpoints. Preserving latest: $last_checkpoint_name"

    for checkpoint in "${checkpoints[@]}"; do
        checkpoint_name=$(basename "$checkpoint")
        step=${checkpoint_name#global_step_}

        if [ "$checkpoint" = "$last_checkpoint" ]; then
            echo "  [$checkpoint_name] Keeping latest checkpoint."
            continue
        fi

        if ! [[ "$step" =~ ^[0-9]+$ ]]; then
            echo "  [$checkpoint_name] Step is not numeric, skipping for safety."
            continue
        fi

        if (( step % retention_interval != 0 )); then
            echo "  [$checkpoint_name] Step $step is not divisible by $retention_interval."
            total_checkpoint_dirs_to_delete=$((total_checkpoint_dirs_to_delete + 1))

            if [ "$DRY_RUN" -eq 0 ]; then
                rm -rf "$checkpoint"
                echo "    Deleted checkpoint directory."
            else
                echo "    Would delete checkpoint directory."
            fi

            continue
        fi

        mapfile -t optim_files < <(find "$checkpoint/actor" -name "optim*" 2>/dev/null)

        if [ ${#optim_files[@]} -eq 0 ]; then
            echo "  [$checkpoint_name] Keeping checkpoint. No optim files found."
            continue
        fi

        size=$(du -ch "${optim_files[@]}" 2>/dev/null | tail -1 | cut -f1)
        echo "  [$checkpoint_name] Keeping checkpoint. Found ${#optim_files[@]} optim files ($size)."
        total_optim_files_to_delete=$((total_optim_files_to_delete + ${#optim_files[@]}))

        if [ "$DRY_RUN" -eq 0 ]; then
            for optim_file in "${optim_files[@]}"; do
                rm -f "$optim_file"
                echo "    Deleted: $(basename "$optim_file")"
            done
        else
            for optim_file in "${optim_files[@]}"; do
                echo "    Would delete: $(basename "$optim_file")"
            done
        fi
    done

    echo ""
done

echo "============================================"
echo "Summary"
echo "============================================"
echo "Total checkpoint directories to delete: $total_checkpoint_dirs_to_delete"
echo "Total optim files to delete: $total_optim_files_to_delete"
if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "This was a DRY RUN. No files were actually deleted."
    echo "To actually delete files, run: DRY_RUN=0 $0"
else
    echo ""
    echo "Cleanup complete!"
fi
