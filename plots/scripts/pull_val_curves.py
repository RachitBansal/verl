"""Pull full AIME validation curves for the plain-GRPO n=16 batch sweep (KL 1e-3 and 1e-2).

Output: JSON list, one entry per run:
  {name, id, bsz, lr, kl, fixed, created, steps: [...], vals: [...]}
where steps/vals are the validation checkpoints (val-core/aime-1983-2024/reward/mean@1).
Downsample runs and n != 16 are excluded. Feeds build_steps_vs_bsz_interactive.py, which
recomputes steps-to-target for any accuracy threshold client-side.

Usage: python pull_val_curves.py out.json
"""
import re, sys, json, os
import wandb

INCR = "--incremental" in sys.argv
OUT = [a for a in sys.argv[1:] if a != "--incremental"][0]
V = "val-core/aime-1983-2024/reward/mean@1"
KLS = (1e-3, 1e-2)

api = wandb.Api(timeout=120)
keep = {}
if INCR and os.path.exists(OUT):
    keep = {d["id"]: d for d in json.load(open(OUT)) if d.get("state") not in (None, "running")}
out = []
for r in api.runs("harvardml/grpo_on_policy_cbs", per_page=500):
    if r.name.startswith("downsample"):
        continue
    m_n = re.match(r"^n(\d+)_bsz(\d+)", r.name)
    if not m_n or int(m_n[1]) != 16:
        continue
    a = r.config.get("actor_rollout_ref", {}).get("actor", {})
    kl = a.get("kl_loss_coef")
    if kl is None or not any(abs(kl - k) <= 1e-12 for k in KLS):
        continue
    m_lr = re.search(r"lr([0-9.]+e-?\d+)", r.name)
    lr = float(m_lr.group(1)) if m_lr else a.get("optim", {}).get("lr")
    if r.id in keep and r.state != "running":   # --incremental: terminal run already pulled
        out.append(keep[r.id]); continue
    rows = sorted((x["_step"], x[V]) for x in r.scan_history(keys=["_step", V]) if x.get(V) is not None)
    if not rows:
        continue
    out.append(dict(name=r.name, id=r.id, state=r.state, bsz=int(m_n[2]), lr=lr, kl=kl,
                    fixed="updated_scaling" in r.name, created=str(r.created_at)[:10],
                    steps=[int(s) for s, _ in rows], vals=[round(v, 4) for _, v in rows]))
    print(f"{r.name:50s} bsz={out[-1]['bsz']:5d} lr={lr:.0e} kl={kl:g} n_val={len(rows):3d} max={max(v for _, v in rows):.3f}", flush=True)

out.sort(key=lambda d: (d["kl"], d["bsz"], d["lr"], d["created"]))
with open(OUT, "w") as f:
    json.dump(out, f)
print("wrote", OUT, len(out), "runs")
