set -euo pipefail

####################
# Sweep over BSZ values for fully on-policy GRPO.
# Doubles LR with each BSZ doubling (linear scaling rule), starting from BSZ=4, LR=1e-6.
# Usage: N=16 bash sweep_on_policy.sh
####################

BSZ_VALUES=(4   8     16    32    64     128    256)
LR_VALUES=(1e-6 2e-6  4e-6  8e-6  1.6e-5 3.2e-5 6.4e-5)

N=${N:-16}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

for i in "${!BSZ_VALUES[@]}"; do
    bsz=${BSZ_VALUES[$i]}
    lr=${LR_VALUES[$i]}
    echo "=========================================="
    echo "Submitting N=${N} BSZ=${bsz} LR=${lr}"
    echo "=========================================="
    sbatch \
        --job-name="cbs_n${N}_bsz${bsz}" \
        --export=ALL,BSZ=${bsz},N=${N},LR=${lr} \
        "${SCRIPT_DIR}/on_policy.sh"
done
