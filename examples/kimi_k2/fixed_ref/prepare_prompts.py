"""Prompt files for the fixed-pi_ref-dataset study (Phase2a_Fixed_PiRef_Dataset_Plan.md §2).

Draws, with a seed, disjoint prompt sets from the DAPO parquet after filtering prompts longer than
--max-prompt-length chat-template tokens (so the trainer's own filter is a no-op and the dataloader's first
batch is exactly the bank):

  bank_prompts.parquet   B training prompts followed by H held-out fresh prompts (trainer: data.shuffle=False,
                         train_batch_size=B, kimi_holdout_prompts=H)
  monitor_seen.parquet   M of the training prompts, data_source math_dapo_bank   (on-policy reward on seen prompts)
  monitor_fresh.parquet  M other prompts,            data_source math_dapo_fresh  (on-policy reward on unseen prompts)
  indices.json           the parquet row indices of every set

Usage: python prepare_prompts.py --out DIR [--bank 4096 --holdout 256 --monitor 512 --seed 0]
"""

import argparse
import json
import os

import pandas as pd
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--dapo", default="/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/data/dapo.parquet")
ap.add_argument("--model", default="/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--bank", type=int, default=4096)
ap.add_argument("--holdout", type=int, default=256)
ap.add_argument("--monitor", type=int, default=512)
ap.add_argument("--max-prompt-length", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

df = pd.read_parquet(args.dapo)
tok = AutoTokenizer.from_pretrained(args.model)
lens = [len(tok.apply_chat_template(list(p), add_generation_prompt=True, tokenize=True)) for p in df["prompt"]]
ok = [i for i, n in enumerate(lens) if n <= args.max_prompt_length]
print(f"{len(df)} prompts, {len(ok)} within {args.max_prompt_length} tokens")
rng = __import__("numpy").random.default_rng(args.seed)
perm = rng.permutation(ok)
need = args.bank + args.holdout + args.monitor
assert len(perm) >= need, f"need {need} prompts, have {len(perm)}"
train_idx = perm[: args.bank].tolist()
hold_idx = perm[args.bank : args.bank + args.holdout].tolist()
fresh_idx = perm[args.bank + args.holdout : need].tolist()
seen_idx = train_idx[: args.monitor]

os.makedirs(args.out, exist_ok=True)
df.iloc[train_idx + hold_idx].reset_index(drop=True).to_parquet(os.path.join(args.out, "bank_prompts.parquet"))
seen = df.iloc[seen_idx].reset_index(drop=True).copy()
seen["data_source"] = "math_dapo_bank"
seen.to_parquet(os.path.join(args.out, "monitor_seen.parquet"))
fresh = df.iloc[fresh_idx].reset_index(drop=True).copy()
fresh["data_source"] = "math_dapo_fresh"
fresh.to_parquet(os.path.join(args.out, "monitor_fresh.parquet"))
with open(os.path.join(args.out, "indices.json"), "w") as f:
    json.dump(
        {
            "seed": args.seed,
            "max_prompt_length": args.max_prompt_length,
            "train": train_idx,
            "holdout_prompt": hold_idx,
            "monitor_seen": seen_idx,
            "monitor_fresh": fresh_idx,
        },
        f,
    )
print(f"wrote {args.out}: bank {len(train_idx)}+{len(hold_idx)}, monitor seen {len(seen_idx)}, fresh {len(fresh_idx)}")
