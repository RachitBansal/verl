"""GPU hours spent on the runs behind the steps-to-50 LR grids, per sweep.

Sweeps and admission (same rules as the paper figures; every LR tried counts, since the grids use them all):
  K=16 batch sweep      n16_bsz*, KL 1e-3 and 1e-2, pre-fix and fixed code (Figure 1 admits pre-fix runs at K = 16)
  K=64 batch sweep      n64_bsz*, KL 1e-3, fixed code only
  B=128 rollout sweep   n*_bsz128, KL 1e-3; pre-fix only for K <= 16, fixed code for K >= 32
  B=1024 rollout sweep  n*_bsz1024, KL 1e-3, fixed code (running jobs count their time so far)
  downsampling          downsample_n64_dsk*_bsz128, KL 1e-3, fixed code (the B = 128 downsampling figures)
Shared cells (K=16 at B=128 and B=1024; K=64 at B=1024) are counted once, under the batch sweep.

Hours = wandb runtime x 4 GPUs. Resumed / requeued segments of one run name overlap in steps; a segment's time is
scaled by the share of its step range not already covered by an earlier segment, so redone steps are not double
counted ("productive"). Segments with no validation rows (crashed before the first eval) are listed separately as
"failed attempts" and excluded from the productive total.

Usage: python gpu_hours_sweeps.py csv/gpu_hours_sweeps.csv csv/val_curves_n16.json csv/val_curves_n64.json csv/val_curves_bsz128.json csv/val_curves_bsz1024.json
"""
import sys, re, json
import wandb, pandas as pd

OUT, *curve_files = sys.argv[1:]
GPUS = 4
KLS = {1e-3, 1e-2}
steps_by_id = {}
for f in curve_files:
    for r in json.load(open(f)):
        if r["steps"]:
            steps_by_id[r["id"]] = (min(r["steps"]), max(r["steps"]))   # val_curves_bsz128.json also holds the downsampling runs


def sweep_of(name, kl, fixed):
    if kl is None or not any(abs(kl - k) < 1e-12 for k in KLS):
        return None
    if name.startswith("downsample"):
        return "downsampling (B=128)" if (fixed and abs(kl - 1e-3) < 1e-12 and re.match(r"^downsample_n64_dsk\d+_bsz128", name)) else None
    m = re.match(r"^n(\d+)_bsz(\d+)(?:_|$)", name)
    if not m:
        return None
    K, B = int(m.group(1)), int(m.group(2))
    kl3 = abs(kl - 1e-3) < 1e-12
    if K == 16:
        return "K=16 batch sweep"                                   # includes B=128 and B=1024 shared cells
    if K == 64 and fixed and kl3:
        return "K=64 batch sweep"                                   # includes the B=1024 shared cell
    if B == 128 and kl3 and ((K <= 16) or fixed):
        return "B=128 rollout sweep"
    if B == 1024 and kl3 and fixed:
        return "B=1024 rollout sweep"
    return None


api = wandb.Api(timeout=120)
rows = []
for r in api.runs("harvardml/grpo_on_policy_cbs", per_page=500):
    a = r.config.get("actor_rollout_ref", {}).get("actor", {}); kl = a.get("kl_loss_coef")
    fixed = "updated_scaling" in r.name
    sw = sweep_of(r.name, kl, fixed)
    if sw is None:
        continue
    rt = r.summary.get("_runtime") or r.summary.get("_wandb", {}).get("runtime") or 0
    lo, hi = steps_by_id.get(r.id, (None, None))
    rows.append(dict(sweep=sw, id=r.id, name=r.name, user=r.user.username if r.user else "", state=r.state, created=str(r.created_at)[:16],
                     kl=kl, fixed=fixed, runtime_h=rt / 3600, gpu_h=rt / 3600 * GPUS, first_step=lo, last_step=hi))
df = pd.DataFrame(rows)

# overlap handling per run name: earlier-created segments claim their step range first
df["productive_gpu_h"] = 0.0; df["failed_attempt"] = df["first_step"].isna()
for name, g in df.groupby("name"):
    covered = []
    for i in g.sort_values("created").index:
        lo, hi = df.loc[i, "first_step"], df.loc[i, "last_step"]
        if pd.isna(lo):
            continue
        span = max(hi - lo, 1); new = span
        for a, b in covered:
            o = max(0, min(hi, b) - max(lo, a)); new -= o
        new = max(new, 0)
        df.loc[i, "productive_gpu_h"] = df.loc[i, "gpu_h"] * new / span
        covered.append((lo, hi))
df.to_csv(OUT, index=False)

order = ["K=16 batch sweep", "K=64 batch sweep", "B=128 rollout sweep", "B=1024 rollout sweep", "downsampling (B=128)"]
print(f"{'sweep':22s} {'runs':>5s} {'segments':>9s} {'productive GPU-h':>17s} {'all segments GPU-h':>19s} {'failed attempts GPU-h':>22s} {'running now':>12s}")
tot = dict(p=0, a=0, f=0)
for sw in order:
    g = df[df["sweep"] == sw]
    if g.empty:
        continue
    p, a, f = g["productive_gpu_h"].sum(), g["gpu_h"].sum(), g.loc[g["failed_attempt"], "gpu_h"].sum()
    tot["p"] += p; tot["a"] += a; tot["f"] += f
    print(f"{sw:22s} {g['name'].nunique():5d} {len(g):9d} {p:17,.0f} {a:19,.0f} {f:22,.0f} {int((g['state'] == 'running').sum()):12d}")
print(f"{'total':22s} {df['name'].nunique():5d} {len(df):9d} {tot['p']:17,.0f} {tot['a']:19,.0f} {tot['f']:22,.0f}")
print("\nK=16 batch sweep by KL: " + ",  ".join(f"KL {kl:g}: {g['productive_gpu_h'].sum():,.0f} GPU-h over {g['name'].nunique()} runs" for kl, g in df[df['sweep'] == 'K=16 batch sweep'].groupby('kl')))
print("K=16 batch sweep by code: " + ",  ".join(f"{'fixed' if fx else 'pre-fix'}: {g['productive_gpu_h'].sum():,.0f} GPU-h over {g['name'].nunique()} runs" for fx, g in df[df['sweep'] == 'K=16 batch sweep'].groupby('fixed')))
print("by user: " + ",  ".join(f"{u}: {g['productive_gpu_h'].sum():,.0f}" for u, g in df.groupby("user")))
print("wrote", OUT, len(df), "segments")
