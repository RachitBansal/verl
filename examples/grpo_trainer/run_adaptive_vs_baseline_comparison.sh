#!/usr/bin/env bash
# ============================================================================
# Adaptive Rollout vs Fixed Rollout Comparison Experiment
# ============================================================================
#
# This script runs a controlled comparison between:
# 1. Adaptive rollout: Variable rollouts up to max_n, stopping when positive found
# 2. Baseline: Fixed n=5 rollouts with larger batch for fair compute comparison
#
# The goal is to evaluate whether adaptive rollouts (focusing on hard examples)
# lead to better sample efficiency compared to fixed rollouts.
#
# Author: Sunny + Claude
# ============================================================================

set -e

# ============================================================================
# Common Configuration
# ============================================================================
export STEP_NUM=${STEP_NUM:-22000}
export OLMO_CHECKPOINT=${OLMO_CHECKPOINT:-"/n/netscratch/dam_lab/Everyone/rl_pretrain/OLMo2-1B-stage1-50B/step${STEP_NUM}-hf"}
export WANDB_ENTITY="harvardml"

# Dataset
DATA_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"

# Output
OUTPUT_DIR="/n/netscratch/dam_lab/Everyone/rl_pretrain/experiments"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Experiment Selection
# ============================================================================
# Run with: ./run_adaptive_vs_baseline_comparison.sh [adaptive|baseline|both]

MODE=${1:-"both"}

run_adaptive() {
    echo "============================================================"
    echo "Running ADAPTIVE ROLLOUT Experiment"
    echo "  Max rollouts per sample: 128"
    echo "  Stopping early when positive found"
    echo "============================================================"

    export MAX_N=128
    export ROLLOUTS_PER_BATCH=8
    export POSITIVE_THRESHOLD=0.5
    export MIN_ROLLOUTS=2
    export TRAIN_BATCH_SIZE=32
    export PPO_MINI_BATCH_SIZE=32

    bash "${SCRIPT_DIR}/run_adaptive_rollout_experiment.sh"
}

run_baseline() {
    echo "============================================================"
    echo "Running BASELINE (Fixed n=5) Experiment"
    echo "  Fixed 5 rollouts per sample"
    echo "  Larger batch size for fair compute comparison"
    echo "============================================================"

    export TRAIN_BATCH_SIZE=512

    bash "${SCRIPT_DIR}/run_baseline_fixed_n5.sh"
}

case "$MODE" in
    "adaptive")
        run_adaptive
        ;;
    "baseline")
        run_baseline
        ;;
    "both")
        echo "Running both experiments sequentially..."
        run_adaptive
        echo ""
        echo "Adaptive experiment complete. Starting baseline..."
        echo ""
        run_baseline
        ;;
    *)
        echo "Usage: $0 [adaptive|baseline|both]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Experiment(s) complete!"
echo "Check wandb project 'rl_pretrain_adaptive' for results"
echo "============================================================"

