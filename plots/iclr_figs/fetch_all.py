#!/usr/bin/env python3
"""Fetch per-step trajectories for every run in harvardml/grpo_on_policy_cbs.
Single-key scans. Long CSV: run,id,state,step,<metric columns>."""
import csv, re, sys, wandb
PROJECT = "harvardml/grpo_on_policy_cbs"
OUT = sys.argv[1]
FILTER = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None   # optional regex on run name
VAL_KEYS = ["val-core/aime-1983-2024/reward/mean@1", "val-core/data/aime-1983-2024/acc/mean@1"]
OTHER = {"resp_len": "response_length/mean", "prompt_len": "prompt_length/mean",
         "reward": "critic/rewards/mean", "grad_norm": "actor/grad_norm", "kl_loss": "actor/kl_loss"}
def scan_one(run, key):
    out = {}
    try:
        for h in run.scan_history(keys=["_step", key], page_size=5000):
            v = h.get(key); s = h.get("_step")
            if v is not None and s is not None: out[int(s)] = v
    except Exception as e:
        print(f"    scan err {key}: {e}", file=sys.stderr)
    return out
api = wandb.Api(timeout=120)
runs = [r for r in api.runs(PROJECT, per_page=500) if FILTER is None or FILTER.search(r.name)]
print(f"Fetching {len(runs)} runs...", file=sys.stderr)
fields = ["run", "id", "state", "created", "step", "aime_mean1"] + list(OTHER)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for i, r in enumerate(runs):
        summ = r.summary
        vk = next((k for k in VAL_KEYS if k in summ), None)
        val = scan_one(r, vk) if vk else {}
        oth = {n: (scan_one(r, k) if k in summ else {}) for n, k in OTHER.items()}
        steps = sorted(set(val) | set().union(*[set(d) for d in oth.values()]))
        for s in steps:
            row = {"run": r.name, "id": r.id, "state": r.state, "created": str(r.created_at), "step": s,
                   "aime_mean1": val.get(s)}
            for n in OTHER: row[n] = oth[n].get(s)
            w.writerow(row)
        f.flush()
        print(f"[{i+1}/{len(runs)}] {r.name} ({r.state}): {len(val)} val, {len(steps)} rows", file=sys.stderr)
print("DONE", file=sys.stderr)
