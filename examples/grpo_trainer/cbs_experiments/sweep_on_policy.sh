set -euo pipefail

####################
# Sweep over BSZ values for fully on-policy GRPO
# Usage: N=16 bash sweep_on_policy.sh
####################

BSZ_VALUES=(4 8)
N=${N:-16}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

for bsz in "${BSZ_VALUES[@]}"; do
    echo "=========================================="
    echo "Submitting N=${N} BSZ=${bsz}"
    echo "=========================================="
    sbatch \
        --job-name="cbs_n${N}_bsz${bsz}" \
        --export=ALL,BSZ=${bsz},N=${N} \
        "${SCRIPT_DIR}/on_policy.sh"
done
