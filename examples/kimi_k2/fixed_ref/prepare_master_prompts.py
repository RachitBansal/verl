"""Prompt file of the master bank: every DAPO prompt whose chat-template length is within --max-prompt-length, in
DAPO file order (so the trainer's own length filter is a no-op and dataset row i = master prompt row i).

  <out>/master_prompts.parquet   the kept DAPO rows (columns unchanged; extra_info.index = DAPO row)
  <out>/master_prompts.json      {"dapo_rows": [...], "max_prompt_length": ..., "num_dapo": ...}

Usage: python prepare_master_prompts.py --out MASTER_DIR/prompts [--limit N]   (--limit: first N kept prompts, smoke)
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
ap.add_argument("--max-prompt-length", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

df = pd.read_parquet(args.dapo)
tok = AutoTokenizer.from_pretrained(args.model)
lens = [len(tok.apply_chat_template(list(p), add_generation_prompt=True, tokenize=True)) for p in df["prompt"]]
keep = [i for i, n in enumerate(lens) if n <= args.max_prompt_length]
if args.limit > 0:
    keep = keep[: args.limit]
print(f"{len(df)} DAPO prompts, keeping {len(keep)} (<= {args.max_prompt_length} tokens)")
idx_ok = all(int(df.iloc[i]["extra_info"]["index"]) == i for i in keep)
print(f"extra_info.index == DAPO row for all kept prompts: {idx_ok}")
os.makedirs(args.out, exist_ok=True)
df.iloc[keep].reset_index(drop=True).to_parquet(os.path.join(args.out, "master_prompts.parquet"))
with open(os.path.join(args.out, "master_prompts.json"), "w") as f:
    json.dump(
        {
            "dapo": args.dapo,
            "num_dapo": len(df),
            "max_prompt_length": args.max_prompt_length,
            "dapo_rows": keep,
            "extra_info_index_is_dapo_row": idx_ok,
        },
        f,
    )
print(f"wrote {args.out}/master_prompts.parquet")
