#!/usr/bin/env bash
# Kimi-K2 grid: B = 4096 prompts, n = 16 rollouts, tau = 0.03, 100 outer steps,
#   MINI in {1024, 256} prompts per optimizer step (= 4 / 16 inner steps per rollout batch)
#   LR   in {1e-5, 3e-6, 3e-5}
# 6 jobs, 4 GPUs (H100/H200) + 400 GB host memory each, 2-day limit. Resubmitting the same job after a
# time-out resumes from the last checkpoint (saved every 20 steps) in OUTPUT_DIR/<run name>/.
#
# Usage:
#   ACCOUNT=<slurm account> PARTITION=<gpu partition> OUTPUT_DIR=<writable dir> bash launch_b4096_grid.sh [--dry-run]
# Optional: GPU_ARG="--gpus-per-node=nvidia_h200:4" (mixed partitions), MINIS="1024", LRS="1e-5 3e-5" (subset),
#           WANDB_ENTITY / PROJECT (default harvardml / kimi_k2_cbs), MEM (400G), TIME (2-00:00:00).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# VERL_ROOT: the shared snapshot keeps verl/ next to this file; in the git repo this file lives in
# examples/kimi_k2/b4096_grid/ and the repo root is three levels up.
if [ -z "${VERL_ROOT:-}" ]; then
    if [ -d "$HERE/verl" ]; then VERL_ROOT=$HERE/verl; else VERL_ROOT=$(cd "$HERE/../../.." && pwd); fi
fi
export VERL_ROOT
export OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a directory you can write (checkpoints: ~10 GB per run)}
ACCOUNT=${ACCOUNT:?set ACCOUNT to your slurm account (e.g. kempner_<lab>)}
PARTITION=${PARTITION:?set PARTITION (e.g. kempner_h200, kempner_h100)}
GPU_ARG=${GPU_ARG:---gpus-per-node=4}
MEM=${MEM:-400G}; TIME=${TIME:-2-00:00:00}
LOGDIR=${LOGDIR:-$OUTPUT_DIR/slurm_logs}; mkdir -p "$LOGDIR"
MINIS=${MINIS:-"1024 256"}; LRS=${LRS:-"1e-5 3e-6 3e-5"}
DRY=${1:-}
for MINI in $MINIS; do for LR in $LRS; do
    name="k2_b4096_mini${MINI}_lr${LR}"
    echo "== $name"
    cmd="BSZ=4096 N=16 MINI=$MINI LR=$LR TAU=0.03 TOTAL_STEPS=100 TEST_FREQ=5 SAVE_FREQ=20 TOTAL_EPOCHS=40 HOLDOUT_PROMPTS=128 TAG=hp128 \
VERL_ROOT=$VERL_ROOT OUTPUT_DIR=$OUTPUT_DIR \
sbatch -J $name -A $ACCOUNT -p $PARTITION $GPU_ARG --mem=$MEM --time=$TIME -o $LOGDIR/%x-%j.out -e $LOGDIR/%x-%j.err \
$VERL_ROOT/examples/kimi_k2/kimi_k2_b512.sh"
    echo "$cmd"
    if [ "$DRY" != "--dry-run" ]; then eval "$cmd"; fi
done; done
