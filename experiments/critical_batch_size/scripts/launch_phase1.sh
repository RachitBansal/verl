#!/usr/bin/env bash
# Launch Phase 1 CBS experiments as a SLURM array job.
# Usage: bash launch_phase1.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${VERL_DIR}"
mkdir -p logs

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "============================================================"
echo "CBS Phase 1: Vary n_prompts in {32,64,128,256,512,1024}"
echo "  n_rollouts = 8 (fixed)"
echo "  Model: OLMo-2-0425-1B-SFT"
echo "  Dataset: OpenMathInstruct2 (full)"
echo "  GPUs: 4 x H100 per run"
echo "  Array jobs: 6"
echo "============================================================"

echo ""
echo "Planned runs:"
for i in 0 1 2 3 4 5; do
    NP_ARRAY=(32 64 128 256 512 1024)
    NP=${NP_ARRAY[$i]}
    TOTAL=$((NP * 8))
    echo "  [$i] n_prompts=${NP}, n_rollouts=8, total_batch=${TOTAL}"
done

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit: sbatch experiments/critical_batch_size/scripts/cbs_phase1.sbatch"
    echo "Exiting without submitting."
    exit 0
fi

echo "Submitting SLURM array job..."
sbatch experiments/critical_batch_size/scripts/cbs_phase1.sbatch
echo "Done. Monitor with: squeue -u \$USER -n cbs-p1"
