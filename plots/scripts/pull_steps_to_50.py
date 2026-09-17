"""Pull val-core histories for non-downsample runs and compute steps to 50%.

Usage: python pull_steps_to_50.py out.csv [kl1,kl2,...]
Default KL filter is 1e-3; pass e.g. "1e-3,1e-2" to keep several coefficients.
"""
import re, json, sys
import wandb
import pandas as pd

OUT = sys.argv[1]
KLS = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1e-3").split(",")]
METRIC = "val-core/aime-1983-2024/reward/mean@1"
THRESHES = [0.5, 0.55, 0.6, 0.65, 0.7]   # one steps_to_<pct> column per threshold

api = wandb.Api(timeout=120)
runs = api.runs("harvardml/grpo_on_policy_cbs")

rows = []
for r in runs:
    if "downsample" in r.name:
        continue
    a = r.config.get("actor_rollout_ref", {}).get("actor", {})
    kl = a.get("kl_loss_coef")
    if kl is None or not any(abs(kl - k) <= 1e-12 for k in KLS):
        continue
    m_n = re.search(r"^n(\d+)_", r.name)
    m_b = re.search(r"bsz(\d+)", r.name)
    m_lr = re.search(r"lr([0-9.]+e-?\d+)", r.name)
    n = int(m_n.group(1)) if m_n else None
    bsz = int(m_b.group(1)) if m_b else None
    lr = float(m_lr.group(1)) if m_lr else None
    cfg_lr = a.get("optim", {}).get("lr")

    hist = r.history(keys=[METRIC], pandas=True, samples=100000)
    firsts = {f"steps_to_{int(round(t * 100))}": None for t in THRESHES}
    if hist is None or len(hist) == 0 or METRIC not in hist:
        n_val, max_val, last_step = 0, None, None
    else:
        hist = hist.dropna(subset=[METRIC]).sort_values("_step")
        n_val = len(hist)
        max_val = float(hist[METRIC].max())
        last_step = int(hist["_step"].max())
        for t in THRESHES:
            hit = hist[hist[METRIC] >= t]
            firsts[f"steps_to_{int(round(t * 100))}"] = int(hit["_step"].iloc[0]) if len(hit) else None
    rows.append(dict(id=r.id, name=r.name, state=r.state, created=str(r.created_at),
                     n=n, bsz=bsz, lr=lr, cfg_lr=cfg_lr, kl=kl,
                     n_val_points=n_val, max_val=max_val, last_val_step=last_step, **firsts))
    print(f"{r.id} {r.name:50s} n={n} bsz={bsz} lr={lr} max={max_val} last={last_step} "
          f"first50={firsts['steps_to_50']} first60={firsts['steps_to_60']}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("wrote", OUT, len(df))
