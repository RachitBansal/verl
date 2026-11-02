"""Choose one run's bank from the master bank (README.md, "Master bank"): which master prompts are trained on,
which are held out, and which of the n stored samples each role uses.  Writes

  <out>/selection.json        read by the trainer (trainer.fixed_dataset.master_selection) and by launch.sh
  <out>/monitor_seen.parquet  first --monitor training prompts, data_source math_dapo_bank (on-policy val)
  <out>/monitor_fresh.parquet --monitor prompts used by no role, data_source math_dapo_fresh

Roles (prompt rows = rows of the master prompt file):
  train           --train prompts, samples [off, off+k)                   (kimi_holdout 0)
  holdout_prompt  --holdout-prompts other prompts, samples [off, off+k)   (kimi_holdout 1, fresh prompts)
  holdout_sample  first --holdout-sample-prompts training prompts, samples [off+k, off+2k)
                                                                          (kimi_holdout 2, fresh samples)
Prompts are a seeded permutation of the master prompts; --train 0 = every prompt not held out / monitored.
--same-prompts-as <prompts dir/indices.json> reuses an old bank's prompt sets (DAPO rows), e.g. to rerun the
4096-prompt cells from the master bank.

Usage: python prepare_master_selection.py --master MASTER_DIR --out SEL_DIR [--train 4096 --k 16
       --holdout-prompts 256 --holdout-sample-prompts 256 --monitor 512 --sample-offset 0 --seed 0]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--master", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train", type=int, default=4096)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--holdout-prompts", type=int, default=256)
ap.add_argument("--holdout-sample-prompts", type=int, default=256)
ap.add_argument("--monitor", type=int, default=512)
ap.add_argument("--sample-offset", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--same-prompts-as", default=None)
args = ap.parse_args()

with open(os.path.join(args.master, "build_config.json")) as f:
    cfg = json.load(f)
n = int(cfg["n"])
prompt_file = cfg["prompt_file"]
df = pd.read_parquet(prompt_file)
n_prompts = len(df)
assert n_prompts == cfg["num_prompts"]
off, k = args.sample_offset, args.k
need = off + (2 * k if args.holdout_sample_prompts > 0 else k)
assert need <= n, f"sample offset {off} + {'2k' if args.holdout_sample_prompts else 'k'} = {need} > n = {n}"

if args.same_prompts_as:
    with open(args.same_prompts_as) as f:
        old = json.load(f)
    dapo_to_row = {int(df.iloc[i]["extra_info"]["index"]): i for i in range(n_prompts)}
    to_rows = lambda xs: [dapo_to_row[int(x)] for x in xs]  # noqa: E731
    train = to_rows(old["train"])
    hold = to_rows(old["holdout_prompt"])
    seen = to_rows(old["monitor_seen"])
    fresh = to_rows(old["monitor_fresh"])
    hs = train[: args.holdout_sample_prompts]  # the old banks used the first training prompts too
else:
    perm = np.random.default_rng(args.seed).permutation(n_prompts).tolist()
    hold = perm[: args.holdout_prompts]
    fresh = perm[args.holdout_prompts : args.holdout_prompts + args.monitor]
    rest = perm[args.holdout_prompts + args.monitor :]
    train = rest if args.train <= 0 else rest[: args.train]
    assert len(train) == args.train or args.train <= 0, f"only {len(rest)} prompts left for training"
    seen = train[: args.monitor]
    hs = train[: args.holdout_sample_prompts]

sel = {
    "master_dir": os.path.abspath(args.master),
    "prompt_file": prompt_file,
    "n_master": n,
    "k": k,
    "seed": args.seed,
    "sample_offset": off,
    "train": train,
    "holdout_prompt": hold,
    "holdout_sample": hs,
    "samples": {
        "train": list(range(off, off + k)),
        "holdout_prompt": list(range(off, off + k)),
        "holdout_sample": list(range(off + k, off + 2 * k)) if hs else [],
    },
    "monitor_seen": seen,
    "monitor_fresh": fresh,
    "same_prompts_as": args.same_prompts_as,
}
assert not set(train) & set(hold) and not set(train) & set(fresh) and not set(hold) & set(fresh)
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "selection.json"), "w") as f:
    json.dump(sel, f)
for name, rows, src in (("monitor_seen", seen, "math_dapo_bank"), ("monitor_fresh", fresh, "math_dapo_fresh")):
    part = df.iloc[rows].reset_index(drop=True).copy()
    part["data_source"] = src
    part.to_parquet(os.path.join(args.out, f"{name}.parquet"))
print(
    f"wrote {args.out}: train {len(train)} x k={k} (samples {off}..{off + k - 1}), holdout_prompt {len(hold)}, "
    f"holdout_sample {len(hs)} (samples {off + k}..{off + 2 * k - 1}), monitor seen {len(seen)} / fresh {len(fresh)}"
)
