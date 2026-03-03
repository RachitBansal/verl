#!/bin/bash
# Script to delete optimizer states from all checkpoints except the last one
# for each experiment in the experiments directory.

EXPERIMENTS_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments/cbs/"

# Set to 1 for dry-run (just print what would be deleted), 0 to actually delete
DRY_RUN=${DRY_RUN:-0}

echo "============================================"
echo "Optimizer State Cleanup Script"
echo "============================================"
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Dry run: $DRY_RUN (set DRY_RUN=0 to actually delete)"
echo "============================================"
echo ""

total_files_to_delete=0

# Iterate over all experiment directories
for experiment_dir in "$EXPERIMENTS_DIR"/*/; do
    experiment_name=$(basename "$experiment_dir")
    
    # Find all global_step_* directories and sort them numerically
    checkpoints=($(find "$experiment_dir" -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | \
        sed 's/.*global_step_//' | \
        sort -n | \
        sed "s|^|${experiment_dir}global_step_|"))
    
    # Skip if no checkpoints found
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "[$experiment_name] No checkpoints found, skipping."
        continue
    fi
    
    # Skip if only one checkpoint
    if [ ${#checkpoints[@]} -eq 1 ]; then
        echo "[$experiment_name] Only 1 checkpoint, skipping."
        continue
    fi
    
    # Get the last checkpoint (to preserve)
    last_checkpoint="${checkpoints[-1]}"
    last_checkpoint_name=$(basename "$last_checkpoint")
    
    echo "[$experiment_name] Found ${#checkpoints[@]} checkpoints. Preserving: $last_checkpoint_name"
    
    # Iterate over all checkpoints except the last one
    for ((i=0; i<${#checkpoints[@]}-1; i++)); do
        checkpoint="${checkpoints[$i]}"
        checkpoint_name=$(basename "$checkpoint")
        
        # Find optim files in the actor directory
        optim_files=($(find "$checkpoint/actor" -name "optim*" 2>/dev/null))
        
        if [ ${#optim_files[@]} -eq 0 ]; then
            echo "  [$checkpoint_name] No optim files found."
            continue
        fi
        
        # Calculate size of optim files
        size=$(du -ch "${optim_files[@]}" 2>/dev/null | tail -1 | cut -f1)
        
        echo "  [$checkpoint_name] Found ${#optim_files[@]} optim files ($size)"
        total_files_to_delete=$((total_files_to_delete + ${#optim_files[@]}))
        
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
echo "Total files to delete: $total_files_to_delete"
if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "This was a DRY RUN. No files were actually deleted."
    echo "To actually delete files, run: DRY_RUN=0 $0"
else
    echo ""
    echo "Deletion complete!"
fi
