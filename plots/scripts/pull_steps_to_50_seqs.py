"""Pull val-core histories for ALL runs (plain GRPO and downsample) and compute steps to 50%.

Like pull_steps_to_50.py but keeps downsample runs and adds the columns needed to put
plain and downsampled runs on a common "sequences trained per step" axis:

  downsample  True for downsample_* runs
  dsk         K = rollouts kept per prompt (downsample runs only)
  fixed       True if the run name carries _updated_scaling (post loss-scaling fix)
  seqs        sequences that reach the actor per optimizer step
              = bsz * K (downsample) or bsz * n (plain)

steps_to_<pct> is the first validation step whose AIME mean@1 >= pct, as before.

Usage: python pull_steps_to_50_seqs.py out.csv
"""
import re, sys, os
import wandb
import pandas as pd

INCR = "--incremental" in sys.argv                     # reuse rows of runs that were already terminal last pull
OUT = [a for a in sys.argv[1:] if a != "--incremental"][0]
METRIC = "val-core/aime-1983-2024/reward/mean@1"
THRESHES = [0.5, 0.55, 0.6]

api = wandb.Api(timeout=120)
keep = {}
if INCR and os.path.exists(OUT):
    for row in pd.read_csv(OUT).to_dict("records"):
        if str(row.get("state")) != "running":
            keep[row["id"]] = row
rows = []
for r in api.runs("harvardml/grpo_on_policy_cbs", per_page=500):
    a = r.config.get("actor_rollout_ref", {}).get("actor", {})
    kl = a.get("kl_loss_coef")
    cfg_lr = a.get("optim", {}).get("lr")
    ds = r.name.startswith("downsample_")
    m_n = re.search(r"(?:^|_)n(\d+)_", r.name)
    m_b = re.search(r"bsz(\d+)", r.name)
    m_lr = re.search(r"lr([0-9.]+e-?\d+)", r.name)
    m_k = re.search(r"dsk(\d+)", r.name)
    n = int(m_n.group(1)) if m_n else None
    bsz = int(m_b.group(1)) if m_b else None
    lr = float(m_lr.group(1)) if m_lr else cfg_lr
    k = int(m_k.group(1)) if m_k else None
    fixed = "updated_scaling" in r.name
    per_prompt = k if ds else n
    seqs = bsz * per_prompt if (bsz and per_prompt) else None

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
                     downsample=ds, fixed=fixed, n=n, dsk=k, bsz=bsz, seqs=seqs, lr=lr, cfg_lr=cfg_lr, kl=kl,
                     n_val_points=n_val, max_val=max_val, last_val_step=last_step, **firsts))
    print(f"{r.id} {r.name:62s} seqs={seqs} lr={lr} kl={kl} max={max_val} last={last_step} "
          f"first50={firsts['steps_to_50']}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("wrote", OUT, len(df))
