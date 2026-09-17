set -euo pipefail

####################
# Sweep over BSZ x LR x KL_COEF combinations for fully on-policy GRPO.
# Usage: N=16 bash sweep_on_policy.sh
####################

# Reruns of bsz=64/lr=4e-6 and bsz=128/lr=1e-5 at a different seed, tagged v2
# so they don't share an experiment name / checkpoint dir with the earlier run
# of the same bsz/lr/kl.
BSZ_VALUES=(256 64)
LR_VALUES=(1e-5 6e-6)
KL_VALUES=(1e-2 1e-2)
TAG_VALUES=()

N=${N:-16}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

for i in "${!BSZ_VALUES[@]}"; do
    bsz=${BSZ_VALUES[$i]}
    lr=${LR_VALUES[$i]}
    kl=${KL_VALUES[$i]}
    tag=${TAG_VALUES[$i]:-}
    echo "=========================================="
    echo "Submitting N=${N} BSZ=${bsz} LR=${lr} KL_COEF=${kl} TAG=${tag}"
    echo "=========================================="
    sbatch \
        --job-name="cbs_n${N}_bsz${bsz}" \
        --export=ALL,BSZ=${bsz},N=${N},LR=${lr},KL_COEF=${kl},TAG=${tag} \
        "${SCRIPT_DIR}/on_policy.sh"
done
