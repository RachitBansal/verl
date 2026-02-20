#!/usr/bin/env bash
# Launch Phase 4 CBS scale-up and ablation experiments.
# Usage: bash launch_phase4.sh [--dry-run] [--block 1|2|3|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_DIR}"
mkdir -p logs

DRY_RUN=false
BLOCK="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --block)   BLOCK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "CBS Phase 4: Scale-up & Ablations"
echo "============================================================"

echo ""
echo "Block 1 (array 0-5): MATH dataset with key batch sizes"
echo "  n_prompts in {64, 128, 256, 512, 1024, 2048}, n_rollouts=8"
echo ""
echo "Block 2 (array 6-11): REINFORCE++ advantage estimator"
echo "  Same batch sizes, different advantage normalization"
echo ""
echo "Block 3 (array 12-17): Learning rate sensitivity"
echo "  LR in {5e-7, 1e-6, 5e-6} at n_prompts={128, 512}"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Would submit:"
    if [[ "$BLOCK" == "all" ]]; then
        echo "  sbatch --array=0-17 experiments/critical_batch_size/scripts/cbs_phase4.sbatch"
    elif [[ "$BLOCK" == "1" ]]; then
        echo "  sbatch --array=0-5 experiments/critical_batch_size/scripts/cbs_phase4.sbatch"
    elif [[ "$BLOCK" == "2" ]]; then
        echo "  sbatch --array=6-11 experiments/critical_batch_size/scripts/cbs_phase4.sbatch"
    elif [[ "$BLOCK" == "3" ]]; then
        echo "  sbatch --array=12-17 experiments/critical_batch_size/scripts/cbs_phase4.sbatch"
    fi
    exit 0
fi

echo ""
echo "Submitting..."
if [[ "$BLOCK" == "all" ]]; then
    sbatch --array=0-17 experiments/critical_batch_size/scripts/cbs_phase4.sbatch
elif [[ "$BLOCK" == "1" ]]; then
    sbatch --array=0-5 experiments/critical_batch_size/scripts/cbs_phase4.sbatch
elif [[ "$BLOCK" == "2" ]]; then
    sbatch --array=6-11 experiments/critical_batch_size/scripts/cbs_phase4.sbatch
elif [[ "$BLOCK" == "3" ]]; then
    sbatch --array=12-17 experiments/critical_batch_size/scripts/cbs_phase4.sbatch
fi
echo "Monitor with: squeue -u \$USER | grep cbs"
