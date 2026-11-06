#!/usr/bin/env bash
# Full-batch master-bank grid (README.md "Master bank"): every master prompt except 256 held-out + 512 monitor
# prompts (17,141) x k=16 samples, one full-batch Adam step per step, group-mean baseline, eval every step.
# tau in {1, 0.3, 0.1, 0.03, 0.01} x 5 learning rates, shifted by half a decade for large tau (the target log-ratio
# A/tau shrinks as tau grows while the Adam step does not).  Each cell is submitted twice (kempner_h200 + seas_gpu
# H200) under one job name; the first to start cancels its pending twin (CANCEL_TWINS).  Host memory: ~0.7 TB
# projected for 282k rows (37 GB + 2.4 MB/row), requested 1400G.  Resubmit the same command to resume.
#
#   bash examples/kimi_k2/fixed_ref/launch_full_grid.sh            # from the verl root
#   DRY=1 bash examples/kimi_k2/fixed_ref/launch_full_grid.sh      # print the sbatch commands only
#   PRECISION=tf32 TAUS="1 0.3" bash examples/kimi_k2/fixed_ref/launch_full_grid.sh   # fp32/TF32 actor + tf32 anchor
# PRECISION (README.md "Numerical precision"): bf16 forwards carry ~0.5 nats rms of sequence-level noise, which
# swamps the regression targets at tau >= 0.3; tf32 needs the anchors/tf32 shards (MODE=master_anchor).
# BASELINE=mean|tau_logz, BASELINE_FILE=<npz from compute_master_baselines.py>: b(x) over all pooled samples of the
# prompt (e.g. ${MASTER}/baselines/n512.npz) instead of the k=16 group; job names get _bl{mean,tlogz}N.
set -euo pipefail
MASTER=${MASTER:-/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/master_dapo_n64}
SEL=${SEL:-${MASTER}/selections/full_k16_hp256_hs256_m512_s0}
STEPS=${STEPS:-200}
TAUS=${TAUS:-"1 0.3 0.1 0.03 0.01"}
PRECISION=${PRECISION:-bf16}
BASELINE=${BASELINE:-mean}
BASELINE_FILE=${BASELINE_FILE:-}
if [ -n "${BASELINE_FILE}" ]; then
    [ -f "${BASELINE_FILE}" ] || { echo "missing ${BASELINE_FILE} (compute_master_baselines.py)"; exit 1; }
    BL_N=$(python3 -c "import json; print(json.load(open('${BASELINE_FILE%.npz}.json'))['meta']['n_total'])")
fi
if [ "${PRECISION}" != "bf16" ]; then
    n_sh=$(python3 -c "import json; print(len(json.load(open('${MASTER}/manifest.json'))['shards']))")
    n_done=$(ls "${MASTER}/anchors/${PRECISION}/"shard_*.json 2>/dev/null | wc -l)
    [ "${n_done}" = "${n_sh}" ] || { echo "anchors/${PRECISION}: ${n_done}/${n_sh} shards (MODE=master_anchor first)"; exit 1; }
fi
[ -f "${SEL}/selection.json" ] || { echo "missing ${SEL}/selection.json (prepare_master_selection.py --train 0)"; exit 1; }

declare -A LRS
LRS[1]="1e-7 3e-7 1e-6 3e-6 1e-5"
LRS[0.3]="1e-7 3e-7 1e-6 3e-6 1e-5"
LRS[0.1]="3e-7 1e-6 3e-6 1e-5 3e-5"
LRS[0.03]="3e-7 1e-6 3e-6 1e-5 3e-5"
LRS[0.01]="3e-7 1e-6 3e-6 1e-5 3e-5"

for TAU in ${TAUS}; do
    for LR in ${LRS[$TAU]}; do
        name=k2full_t${TAU}_lr${LR}
        if [ "${PRECISION}" != "bf16" ]; then name=${name}_${PRECISION}; fi
        if [ -n "${BASELINE_FILE}" ]; then
            name=${name}_bl$([ "${BASELINE}" = tau_logz ] && echo tlogz || echo mean)${BL_N}
        elif [ "${BASELINE}" != mean ]; then
            name=${name}_bl${BASELINE}
        fi
        env_vars=(SELECTION="${SEL}" TAU="${TAU}" LR="${LR}" STEPS="${STEPS}" TEST_FREQ=1 VAL_N=4
            SAVE_FREQ=5 CKPT_KEEP=1 HF_SAVE_FREQ=10 HF_DENSE=0 BASELINE="${BASELINE}" BASELINE_FILE="${BASELINE_FILE}"
            CANCEL_TWINS=1 PRECISION="${PRECISION}")
        for part in "-p kempner_h200 -A kempner_dam_lab --gpus-per-node=4" \
            "-p seas_gpu -A dam_lab --gpus-per-node=nvidia_h200:4"; do
            # shellcheck disable=SC2086
            cmd=(env "${env_vars[@]}" sbatch --parsable -J "${name}" --mem=1400G --time=2-00:00:00 ${part}
                examples/kimi_k2/fixed_ref/launch.sh)
            if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; else echo "${name} $(echo ${part} | cut -d' ' -f2) $("${cmd[@]}")"; fi
        done
    done
done
