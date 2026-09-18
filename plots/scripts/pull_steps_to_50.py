"""Pull val-core histories for non-downsample runs and compute steps to 50%.

Usage: python pull_steps_to_50.py out.csv [kl1,kl2,...]
Default KL filter is 1e-3; pass e.g. "1e-3,1e-2" to keep several coefficients.
"""
import re, json, sys, os
import wandb
import pandas as pd

INCR = "--incremental" in sys.argv                     # reuse rows of runs that were already terminal last pull
args = [a for a in sys.argv[1:] if a != "--incremental"]
OUT = args[0]
KLS = [float(x) for x in (args[1] if len(args) > 1 else "1e-3").split(",")]
METRIC = "val-core/aime-1983-2024/reward/mean@1"
THRESHES = [0.5, 0.55, 0.6, 0.65, 0.7]   # one steps_to_<pct> column per threshold

api = wandb.Api(timeout=120)
keep = {}
if INCR and os.path.exists(OUT):
    for row in pd.read_csv(OUT).to_dict("records"):
        if str(row.get("state")) != "running":
            keep[row["id"]] = row
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

    if r.id in keep and r.state != "running":   # --incremental: terminal run already pulled, reuse its row
        rows.append(keep[r.id]); continue
    hist = r.history(keys=[METRIC], pandas=True, samples=100000)
    firsts = {}
    for t in THRESHES:
        firsts[f"steps_to_{int(round(t * 100))}"] = None; firsts[f"steps_to_{int(round(t * 100))}_interp"] = None
    if hist is None or len(hist) == 0 or METRIC not in hist:
        n_val, max_val, last_step = 0, None, None
    else:
        hist = hist.dropna(subset=[METRIC]).sort_values("_step")
        n_val = len(hist)
        max_val = float(hist[METRIC].max())
        last_step = int(hist["_step"].max())
        steps_l, vals_l = hist["_step"].tolist(), hist[METRIC].tolist()
        for t in THRESHES:
            col = f"steps_to_{int(round(t * 100))}"
            i = next((j for j, v in enumerate(vals_l) if v >= t), None)
            if i is None:
                firsts[col] = None; firsts[col + "_interp"] = None
            else:
                firsts[col] = int(steps_l[i])   # first validation checkpoint at/above the threshold
                if i == 0 or vals_l[i] == vals_l[i - 1]:
                    firsts[col + "_interp"] = float(steps_l[i])
                else:   # linear crossing between the two bracketing validation readings
                    s0, v0, s1, v1 = steps_l[i - 1], vals_l[i - 1], steps_l[i], vals_l[i]
                    firsts[col + "_interp"] = float(s0 + (s1 - s0) * (t - v0) / (v1 - v0))
    rows.append(dict(id=r.id, name=r.name, state=r.state, created=str(r.created_at),
                     n=n, bsz=bsz, lr=lr, cfg_lr=cfg_lr, kl=kl,
                     n_val_points=n_val, max_val=max_val, last_val_step=last_step, **firsts))
    print(f"{r.id} {r.name:50s} n={n} bsz={bsz} lr={lr} max={max_val} last={last_step} "
          f"first50={firsts['steps_to_50']} first60={firsts['steps_to_60']}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("wrote", OUT, len(df))
