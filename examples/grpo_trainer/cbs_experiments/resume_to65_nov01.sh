#!/usr/bin/env bash
# Resume two K16 runs toward the 65% AIME target (2026-11-01). Run as cmohri FROM THE REPO ROOT
# (on_policy.sh writes logs/slurm-%j.out relative to the submit dir):
#     bash examples/grpo_trainer/cbs_experiments/resume_to65_nov01.sh
# Each job stops as soon as AIME val >= 0.65. Steps/epoch = floor(17917/BSZ): 1024 -> 17, 4096 -> 4.
set -euo pipefail
S=examples/grpo_trainer/cbs_experiments/on_policy.sh

# 1. B=1024 lr1e-5: auto-resumes from cmohri's own global_step_100 (0.627 @ 170 before the crash).
#    Folder uses the old "kl1e-3" spelling, so KL_COEF must be 1e-3. Cap 30 epochs = 510 steps.
N=16 BSZ=1024 LR=1e-5 KL_COEF=1e-3 STOP_AT=0.65 TEST_FREQ=5 SAVE_FREQ=10 TOTAL_EPOCHS=30 \
  sbatch --time=2-00:00:00 --job-name=cbs_n16_bsz1024_lr1e-5_to65 $S

# (B=8192 lr3e-5 moved to brachit's side: job runs as cbs_n16_bsz8192_lr3e-5_resume. Do NOT launch it here.)

# 3. B=4096 lr2e-5: continues brachit's dense run from global_step_12 (0.466 @ 10) into a new
#    cmohri folder ..._dense_resume. Cap 40 epochs = 160 steps.
#    If it hits the 2-day limit before 65%, resubmit THIS command WITHOUT RESUME_FROM: it then
#    auto-resumes from its own latest checkpoint. Do not chain an automatic follow-up: if the first
#    job died before saving, the follow-up would silently start B=4096 from scratch.
N=16 BSZ=4096 LR=2e-5 KL_COEF=0.001 TAG=dense_resume STOP_AT=0.65 TEST_FREQ=2 SAVE_FREQ=4 TOTAL_EPOCHS=40 NCCL_TIMEOUT=7200 \
  RESUME_FROM=/n/netscratch/dam_lab/Everyone/brachit/cbs_runs/n16_bsz4096_lr2e-5_kl0.001_updated_scaling_dense/global_step_12 \
  sbatch --mem=500G --time=2-00:00:00 --job-name=cbs_n16_bsz4096_lr2e-5_to65 $S
