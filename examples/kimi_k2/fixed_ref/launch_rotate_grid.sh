#!/usr/bin/env bash
# Batch-size (sub-sampling) study (README.md "Sub-sampled training"): every step trains on K fresh samples per
# training prompt (17,141 prompts of the full selection) drawn from the 64-sample master pool (prompts of the
# held-out-sample set: 48, their samples 16-31 stay held out), reshuffled each epoch; targets r - b(x) with b(x) the
# precomputed 512-sample baseline in every cell.  K = 64 is full-batch gradient descent on the 64-sample pool
# objective; K = 4 / 16 are unbiased sub-samples of the same gradient.  Steps are split into update_actor calls of
# <= MAX_UPDATE_ROWS rows that accumulate into one Adam step (bounds host memory; K=64 = 8 calls).
#
#   bash examples/kimi_k2/fixed_ref/launch_rotate_grid.sh          # from the verl root
#   DRY=1 bash examples/kimi_k2/fixed_ref/launch_rotate_grid.sh    # print the sbatch commands only
# Submission order = the order of CELLS (queue priority).  Each cell twice (kempner_h200 + seas_gpu, CANCEL_TWINS).
# Resubmit the same command to resume (the sample schedule is a function of the step); EXTRA_SBATCH adds sbatch
# options, e.g. EXTRA_SBATCH=--dependency=afterany:<job> CELLS=0.03:4 to queue a resume behind a running job.
set -euo pipefail
MASTER=${MASTER:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/master_dapo_n64}
SEL=${SEL:-${MASTER}/selections/full_k16_hp256_hs256_m512_s0}
BASELINE=${BASELINE:-mean}
BASELINE_FILE=${BASELINE_FILE:-${MASTER}/baselines/n512.npz}
LR=${LR:-1e-5}
STEPS=${STEPS:-200}
MAX_UPDATE_ROWS=${MAX_UPDATE_ROWS:-140000}
MEM=${MEM:-900G}
CELLS=${CELLS:-"0.03:4 0.03:64 0.03:16 0.1:4 0.1:64 0.1:16"}   # tau:K
[ -f "${BASELINE_FILE}" ] || { echo "missing ${BASELINE_FILE}"; exit 1; }
BL_N=$(python3 -c "import json; print(json.load(open('${BASELINE_FILE%.npz}.json'))['meta']['n_total'])")
for cell in ${CELLS}; do
    TAU=${cell%%:*}
    K=${cell##*:}
    # checkpoint roughly every 1.5-3 h of wall clock (step: ~17 min at K=4, ~50 min at K=16, ~3 h at K=64)
    case ${K} in 4) SAVE=5; HF=10 ;; 16) SAVE=2; HF=10 ;; *) SAVE=1; HF=5 ;; esac
    name=k2rot_K${K}_t${TAU}_lr${LR}_bl${BASELINE}${BL_N}
    env_vars=(SELECTION="${SEL}" TAU="${TAU}" LR="${LR}" STEPS="${STEPS}" TEST_FREQ=1 VAL_N=4
        SAVE_FREQ=${SAVE} CKPT_KEEP=1 HF_SAVE_FREQ=${HF} HF_DENSE=0 BASELINE="${BASELINE}" BASELINE_FILE="${BASELINE_FILE}"
        ROTATE_K="${K}" MAX_UPDATE_ROWS="${MAX_UPDATE_ROWS}" CANCEL_TWINS=1 PRECISION=bf16)
    for part in "-p kempner_h200 -A kempner_dam_lab --gpus-per-node=4" \
        "-p seas_gpu -A dam_lab --gpus-per-node=nvidia_h200:4"; do
        # shellcheck disable=SC2086
        cmd=(env "${env_vars[@]}" sbatch --parsable -J "${name}" --mem=${MEM} --time=2-00:00:00 ${part} ${EXTRA_SBATCH:-}
            examples/kimi_k2/fixed_ref/launch.sh)
        if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else echo "${name} $(echo ${part} | cut -d' ' -f2) $("${cmd[@]}")"; fi
    done
done
