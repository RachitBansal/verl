#!/usr/bin/env bash
# Sub-sampled K2 runs with K = 4 fresh samples per prompt per step (README.md section 2f), for a (tau, lr) list.
# Each run: all 17,141 training prompts of the master bank x 4 fresh samples per step (69k rows), drawn from the
# 64-sample pool and reshuffled every 16 steps; targets r - b(x) with b(x) the 512-sample mean baseline; frozen
# pi_ref anchor; one Adam step per step; bf16; AIME + 2 DAPO monitor sets evaluated every step; 200 steps.
#
# Usage (from anywhere):
#   ACCOUNT=<slurm account> PARTITION=<gpu partition> OUTPUT_DIR=<writable dir> bash launch_k4_sweep.sh [--dry-run]
# Defaults: CELLS="0.003:1e-5 0.003:3e-5 0.003:3e-6 0.03:3e-5 0.03:3e-6" (tau:lr pairs), GPU_ARG=--gpus-per-node=4,
#   MEM=400G (peak host RSS of a K=4 run is ~245 GB), TIME=2-00:00:00, STEPS=200, HF_SAVE_FREQ=10.
# Optional second partition: ACCOUNT2/PARTITION2/GPU_ARG2 submit a twin of every job under the same name; whichever
#   starts first cancels the other (CANCEL_TWINS).
# A run takes ~16 min/step, so ~175 steps fit in 2 days: re-run the same command after a time-out and the job resumes
# from its last checkpoint in OUTPUT_DIR/<run name>/ (saved every 5 steps) with the same sample schedule.
# wandb: entity harvardml, project kimi_k2_fixed (override with WANDB_ENTITY / PROJECT); run `wandb login` once.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# VERL_ROOT: the shared snapshot keeps verl/ next to this file; in the repo this file is examples/kimi_k2/fixed_ref/
if [ -z "${VERL_ROOT:-}" ]; then
    if [ -d "$HERE/verl" ]; then VERL_ROOT=$HERE/verl; else VERL_ROOT=$(cd "$HERE/../../.." && pwd); fi
fi
export VERL_ROOT
export OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a directory you can write (~80 GB per run: 1 checkpoint + hf snapshots)}
ACCOUNT=${ACCOUNT:?set ACCOUNT to your slurm account (e.g. kempner_<lab>)}
PARTITION=${PARTITION:?set PARTITION (e.g. kempner_h200)}
GPU_ARG=${GPU_ARG:---gpus-per-node=4}
MEM=${MEM:-400G}; TIME=${TIME:-2-00:00:00}; STEPS=${STEPS:-200}; HF_SAVE_FREQ=${HF_SAVE_FREQ:-10}
CELLS=${CELLS:-"0.003:1e-5 0.003:3e-5 0.003:3e-6 0.03:3e-5 0.03:3e-6"}
LOGDIR=${LOGDIR:-$OUTPUT_DIR/slurm_logs}; mkdir -p "$LOGDIR"
DATA=${DATA:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/master_dapo_n64}
SEL=${SEL:-$DATA/selections/full_k16_hp256_hs256_m512_s0}
BASELINE_FILE=${BASELINE_FILE:-$DATA/baselines/n512.npz}
for f in "$SEL/selection.json" "$BASELINE_FILE" "$VERL_ROOT/examples/kimi_k2/fixed_ref/launch.sh"; do
    [ -r "$f" ] || { echo "cannot read $f"; exit 1; }
done
DRY=${1:-}
parts=("-A $ACCOUNT -p $PARTITION $GPU_ARG")
if [ -n "${PARTITION2:-}" ]; then parts+=("-A ${ACCOUNT2:-$ACCOUNT} -p $PARTITION2 ${GPU_ARG2:-$GPU_ARG}"); fi
for cell in $CELLS; do
    TAU=${cell%%:*}; LR=${cell##*:}
    name=k2rot_K4_t${TAU}_lr${LR}_blmean512
    echo "== $name"
    for part in "${parts[@]}"; do
        cmd="SELECTION=$SEL TAU=$TAU LR=$LR STEPS=$STEPS TEST_FREQ=1 VAL_N=4 SAVE_FREQ=5 CKPT_KEEP=1 \
HF_SAVE_FREQ=$HF_SAVE_FREQ HF_DENSE=0 BASELINE=mean BASELINE_FILE=$BASELINE_FILE ROTATE_K=4 MAX_UPDATE_ROWS=140000 \
PRECISION=bf16 CANCEL_TWINS=1 VERL_ROOT=$VERL_ROOT OUTPUT_DIR=$OUTPUT_DIR \
sbatch --parsable -J $name $part --mem=$MEM --time=$TIME -o $LOGDIR/%x-%j.out -e $LOGDIR/%x-%j.err \
$VERL_ROOT/examples/kimi_k2/fixed_ref/launch.sh"
        echo "$cmd"
        if [ "$DRY" != "--dry-run" ]; then echo "submitted: $(eval "$cmd")"; fi
    done
done
