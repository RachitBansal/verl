"""Per-prompt baselines over ALL samples of a master bank and its extensions (same prompt file, disjoint sample
ranges), for trainer.fixed_dataset.baseline_file.

    python compute_master_baselines.py --master <base master_dir> --master <extension master_dir> \
        --out <base master_dir>/baselines/n512.npz [--taus 1 0.3 0.1 0.03 0.01] [--check-overlap 2]

Reads only the shard sidecars (per-prompt pass counts; rewards must be binary).  Writes <out> (npz) with, per
prompt row of the prompt file: ``correct``, ``total`` (N), ``mean`` = correct/N, ``taus`` and ``tau_logz``
(T, P) = tau log((1/N) sum_j exp(r_j/tau)), and ``meta`` (json: sources, settings); plus <out>.json with the same
meta and summary statistics.  A run picks a column with trainer.fixed_dataset.baseline_file=<out> and
fixed_dataset.baseline=mean|tau_logz (tau = the run's kimi_tau).

--check-overlap K loads the first K shards' worth of prompts of every bank and counts responses that occur in two
banks (identical token ids): a non-zero count means an extension repeated the base bank's samples (same seed).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from verl.trainer.ppo import kimi_master_bank as mb  # noqa: E402


def overlap(masters: list[str], prompt_rows: list[int]) -> dict:
    """Responses (token-id tuples) of the given prompt rows found in more than one bank, and within-bank repeats."""
    seen: list[dict[int, list[bytes]]] = []
    for m in masters:
        man = mb.load_manifest(m)
        starts = np.array([s["prompt_row_start"] for s in man["shards"]])
        per_prompt: dict[int, list[bytes]] = {}
        for c in sorted({int(np.searchsorted(starts, p, side="right") - 1) for p in prompt_rows}):
            sh = torch.load(os.path.join(m, "shards", man["shards"][c]["file"]), weights_only=False)
            n, p0, ro = sh["n"], int(sh["prompt_rows"][0]), sh["response_offsets"]
            for p in prompt_rows:
                lp = p - p0
                if 0 <= lp < sh["num_prompts"]:
                    per_prompt[p] = [
                        sh["response_tokens"][ro[r] : ro[r + 1]].numpy().tobytes() for r in range(lp * n, lp * n + n)
                    ]
            del sh
        seen.append(per_prompt)
    cross = within = total = 0
    for p in prompt_rows:
        sets = [set(s[p]) for s in seen]
        for s, lst in zip(sets, (x[p] for x in seen), strict=True):
            within += len(lst) - len(s)
            total += len(lst)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                cross += len(sets[i] & sets[j])
    return {"prompts": len(prompt_rows), "responses": total, "cross_bank_duplicates": cross, "within_bank_duplicates": within}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", action="append", required=True, help="master bank dir (repeat; base bank first)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.03, 0.01])
    ap.add_argument("--check-overlap", type=int, default=0, help="number of prompts to check for repeated samples")
    args = ap.parse_args()

    banks = [mb.load_pass_counts(m) for m in args.master]
    base = banks[0]["build_config"]
    for m, b in zip(args.master, banks, strict=True):
        diff = {
            k: (base.get(k), b["build_config"].get(k))
            for k in mb.BASELINE_KEYS_CHECKED
            if base.get(k) != b["build_config"].get(k)
        }
        assert not diff, f"{m} differs from {args.master[0]} in {diff}"
        assert len(b["pass_counts"]) == base["num_prompts"], f"{m}: {len(b['pass_counts'])} prompts"
    # disjoint sample ranges and distinct seeds
    ranges = sorted((b["sample_offset"], b["sample_offset"] + b["n"], m) for m, b in zip(args.master, banks, strict=True))
    for (s0, e0, m0), (s1, e1, m1) in zip(ranges, ranges[1:]):
        assert e0 <= s1, f"sample ranges overlap: {m0} [{s0},{e0}) and {m1} [{s1},{e1})"
    seeds = [int(b["build_config"].get("rollout_seed", 0)) for b in banks]
    assert len(set(seeds)) == len(seeds), f"banks share a rollout seed: {seeds}"
    # same prompts: dapo index per prompt row where the sidecars store it
    dapo = [b["dapo_index"] for b in banks if b["dapo_index"] is not None]
    for d in dapo[1:]:
        assert np.array_equal(d, dapo[0]), "banks disagree on the dataset index of a prompt row"

    correct = np.sum([b["pass_counts"] for b in banks], axis=0).astype(np.int64)
    total = np.full_like(correct, sum(b["n"] for b in banks))
    mean = correct / total
    taus = np.asarray(args.taus, dtype=np.float64)
    tau_logz = np.stack([mb.tau_logz_from_counts(correct, total, t) for t in taus])
    meta = {
        "sources": [
            {
                "master_dir": os.path.realpath(m),
                "n": b["n"],
                "sample_offset": b["sample_offset"],
                "rollout_seed": int(b["build_config"].get("rollout_seed", 0)),
                "reward_sum": b["reward_sum"],
            }
            for m, b in zip(args.master, banks, strict=True)
        ],
        "n_total": int(total[0]),
        "num_prompts": int(len(correct)),
        "taus": taus.tolist(),
        **{k: base.get(k) for k in mb.BASELINE_KEYS_CHECKED},
    }
    stats = {
        "pass_rate_mean": float(mean.mean()),
        f"pass_at_{int(total[0])}": float((correct > 0).mean()),
        "all_correct_frac": float((correct == total).mean()),
        "zero_or_all_frac": float(((correct == 0) | (correct == total)).mean()),
    }
    for b, m in zip(banks, args.master, strict=True):
        stats[f"pass_at_{b['n']}[{os.path.basename(m.rstrip('/'))}]"] = float((b["pass_counts"] > 0).mean())
        stats[f"pass_rate_mean[{os.path.basename(m.rstrip('/'))}]"] = float(b["pass_counts"].mean() / b["n"])
    if len(banks) > 1:
        p0 = banks[0]["pass_counts"] / banks[0]["n"]
        p1 = (correct - banks[0]["pass_counts"]) / (total - banks[0]["n"])
        stats["pass_rate_corr_base_vs_rest"] = float(np.corrcoef(p0, p1)[0, 1])
        stats["pass_rate_mean_diff_rest_minus_base"] = float((p1 - p0).mean())
    for t, row in zip(taus, tau_logz, strict=True):
        stats[f"tau_logz_mean[tau={t:g}]"] = float(row.mean())
    if args.check_overlap:
        stats["overlap"] = overlap(args.master, list(range(min(args.check_overlap, len(correct)))))
        assert stats["overlap"]["cross_bank_duplicates"] == 0 or len(banks) == 1, f"repeated samples: {stats['overlap']}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, correct=correct, total=total, mean=mean, taus=taus, tau_logz=tau_logz, meta=np.array(json.dumps(meta)))
    os.replace(tmp, args.out)
    mb.atomic_write_json(os.path.splitext(args.out)[0] + ".json", {"meta": meta, "stats": stats})
    print(json.dumps({"out": args.out, **stats}, indent=1))


if __name__ == "__main__":
    main()
