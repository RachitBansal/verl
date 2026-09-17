"""Pull training curves for downsample (N=64 -> K, fixed scaling) vs plain GRPO with n = K rollouts.

Both arms train on 128 * K sequences per step at batch 128 prompts; the only difference is
whether the GRPO advantage baseline was computed from 64 rollouts (downsample) or from K.

For each K in KS this pulls, at KL 1e-3:
  role "downsample"  : downsample_n64_dsk{K}_bsz128_lr{LR}_kl1e-3_updated_scaling
  role "plain"       : n{K}_bsz128_lr{LR}*      (pre-fix scaling; effective LR ~0.86-0.99x nominal
                                                  for these small-n runs, eps/sqrt(v) is small)
  role "plain_fixed" : n{K}_bsz128_lr{LR}_kl1e-3_updated_scaling   (K = 16 only)
at LR in LRS. All runs with >= MIN_STEPS logged training steps are kept (seeds/duplicates too).

Output: long-format CSV with one row per (run, step): train reward critic/score/mean and,
where logged, val-core AIME mean@1. Also prints each run's early grad-norm median so the
pre-fix effective LR can be quoted.

Usage: python pull_curves_downsample_vs_n.py out.csv
"""
import re, sys, statistics as st
import wandb
import pandas as pd

OUT = sys.argv[1]
KS = [2, 4, 8, 16]
LRS = [1e-5, 3e-6]
MIN_STEPS = 200
TRAIN = "critic/score/mean"
VAL = "val-core/aime-1983-2024/reward/mean@1"
EPS, SQRTP = 1e-8, 3.9e4   # Adam eps, sqrt(#params) for Qwen2.5-Math-1.5B

api = wandb.Api(timeout=120)
runs = list(api.runs("harvardml/grpo_on_policy_cbs", per_page=500))


def lr_of(name, cfg):
    m = re.search(r"lr([0-9.]+e-?\d+)", name)
    return float(m.group(1)) if m else cfg


rows, meta = [], []
for r in runs:
    a = r.config.get("actor_rollout_ref", {}).get("actor", {})
    kl = a.get("kl_loss_coef")
    if kl is None or abs(kl - 1e-3) > 1e-12:
        continue
    lr = lr_of(r.name, a.get("optim", {}).get("lr"))
    if lr is None or not any(abs(lr - x) / x < 0.05 for x in LRS):
        continue
    md = re.match(r"^downsample_n64_dsk(\d+)_bsz128_.*updated_scaling", r.name)
    mp = re.match(r"^n(\d+)_bsz128(?:_|$)", r.name)
    if md and int(md[1]) in KS:
        k, role = int(md[1]), "downsample"
    elif mp and int(mp[1]) in KS:
        k, role = int(mp[1]), ("plain_fixed" if "updated_scaling" in r.name else "plain")
    else:
        continue

    hist = list(r.scan_history(keys=["_step", TRAIN, "actor/grad_norm"]))
    if len(hist) < MIN_STEPS:
        print(f"skip {r.name}: {len(hist)} steps", flush=True)
        continue
    val = {x["_step"]: x[VAL] for x in r.scan_history(keys=["_step", VAL]) if x.get(VAL) is not None}
    for x in hist:
        rows.append(dict(run=r.name, id=r.id, role=role, k=k, lr=lr, step=x["_step"],
                         train=x.get(TRAIN), val=val.get(x["_step"])))
    for s, v in val.items():   # val rows at steps with no train row (e.g. step 0)
        if not any(x["_step"] == s for x in hist):
            rows.append(dict(run=r.name, id=r.id, role=role, k=k, lr=lr, step=s, train=None, val=v))

    g = st.median(x["actor/grad_norm"] for x in hist[:50] if x.get("actor/grad_norm") is not None)
    c = 1.0 if "updated_scaling" in r.name else (k / 64 ** 2 if role == "downsample" else 1.0 / k)
    eps_ratio = EPS * SQRTP / g
    meta.append((k, role, lr, r.name, len(hist), g, g / c, eps_ratio, lr / (1 + eps_ratio)))
    print(f"{r.name:56s} role={role:12s} k={k:2d} lr={lr:.0e} steps={len(hist):4d} "
          f"gnorm_logged={g:.2e} true={g / c:.2e} eps/sqrtv={eps_ratio:.3f} effLR={lr / (1 + eps_ratio):.2e}", flush=True)

df = pd.DataFrame(rows).sort_values(["k", "role", "lr", "run", "step"])
df.to_csv(OUT, index=False)
pd.DataFrame(meta, columns=["k", "role", "lr", "run", "steps", "gnorm_logged", "gnorm_true", "eps_over_sqrtv", "eff_lr"]) \
    .to_csv(OUT.replace(".csv", "_runs.csv"), index=False)
print("wrote", OUT, len(df), "rows;", len(meta), "runs")
