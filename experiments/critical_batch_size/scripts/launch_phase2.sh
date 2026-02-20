#!/usr/bin/env bash
# Launch Phase 2 CBS experiments (all three sub-experiments).
# Usage: bash launch_phase2.sh [--dry-run] [--part a|b|c|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${VERL_DIR}"
mkdir -p logs

DRY_RUN=false
PART="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --part)    PART="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "CBS Phase 2: Two-Axis Decomposition"
echo "============================================================"

if [[ "$PART" == "all" || "$PART" == "a" ]]; then
    echo ""
    echo "--- Phase 2A: Vary n_prompts, fixed n_rollouts=8 ---"
    echo "  n_prompts in {32, 64, 128, 256, 512, 1024}"
    echo "  (Same grid as Phase 1, tagged as p2a)"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] sbatch experiments/critical_batch_size/scripts/cbs_phase2a.sbatch"
    else
        sbatch experiments/critical_batch_size/scripts/cbs_phase2a.sbatch
    fi
fi

if [[ "$PART" == "all" || "$PART" == "b" ]]; then
    echo ""
    echo "--- Phase 2B: Fix n_prompts=128, vary n_rollouts ---"
    echo "  n_rollouts in {1, 2, 4, 8, 16, 32, 64}"
    echo "  total_batch in {128, 256, 512, 1024, 2048, 4096, 8192}"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] sbatch experiments/critical_batch_size/scripts/cbs_phase2b.sbatch"
    else
        sbatch experiments/critical_batch_size/scripts/cbs_phase2b.sbatch
    fi
fi

if [[ "$PART" == "all" || "$PART" == "c" ]]; then
    echo ""
    echo "--- Phase 2C: Iso-batch (total=2048), vary decomposition ---"
    echo "  (np,nr): (2048,1) (1024,2) (512,4) (256,8) (128,16) (64,32)"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] sbatch experiments/critical_batch_size/scripts/cbs_phase2c.sbatch"
    else
        sbatch experiments/critical_batch_size/scripts/cbs_phase2c.sbatch
    fi
fi

echo ""
echo "Monitor with: squeue -u \$USER | grep cbs"
