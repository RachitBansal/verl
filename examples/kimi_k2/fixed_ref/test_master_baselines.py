"""CPU test of large-N baselines: a base master bank (n=4) + a baseline-only extension (n=12, sample offset 4,
no reference log-probs) over the same prompts -> compute_master_baselines.py -> lookup_master_baseline equals
the mean / tau log Zhat of all 16 pooled scores per prompt (checked against kimi_fixed_bank.soft_value_baseline).
Run: python test_master_baselines.py"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../.."))
from verl.trainer.ppo import kimi_master_bank as mb  # noqa: E402
from verl.trainer.ppo.kimi_fixed_bank import lookup_master_baseline, soft_value_baseline  # noqa: E402

g = torch.Generator().manual_seed(0)
PAD, Lp, Lr = 7, 6, 8
P_TOTAL, CHUNK = 10, 4
P_RATE = torch.rand(P_TOTAL, generator=g) * torch.tensor([0, 1, 1, 1, 0.05, 1, 1, 1, 1, 1.0])  # prompt 0: never
P_RATE[9] = 1.0  # prompt 9: always solved


def build(d: str, n: int, offset: int, with_ref: bool, seed: int) -> torch.Tensor:
    """Bank of P_TOTAL prompts x n samples in shards of CHUNK prompts; returns its (P_TOTAL, n) scores."""
    cfg = {"model": "m", "prompt_file": "p.parquet", "num_prompts": P_TOTAL, "temperature": 1.0, "top_p": 1.0,
           "top_k": -1, "max_prompt_length": Lp, "max_response_length": Lr, "n": n}
    if offset:
        cfg["sample_offset"] = offset
    if seed:
        cfg["rollout_seed"] = seed
    os.makedirs(d, exist_ok=True)
    mb.atomic_write_json(os.path.join(d, "build_config.json"), cfg)
    all_scores = []
    n_chunks = (P_TOTAL + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        rows = list(range(c * CHUNK, min(P_TOTAL, (c + 1) * CHUNK)))
        P, R = len(rows), len(rows) * n
        prompts = torch.full((R, Lp), PAD)
        prompts[:, -3:] = torch.tensor(rows).repeat_interleave(n)[:, None] + 100
        pmask = torch.zeros((R, Lp), dtype=torch.int64)
        pmask[:, -3:] = 1
        rlen = torch.randint(1, Lr + 1, (R,), generator=g)
        rmask = (torch.arange(Lr)[None] < rlen[:, None]).long()
        resp = torch.where(rmask.bool(), torch.randint(200, 300, (R, Lr), generator=g), torch.full((R, Lr), PAD))
        score = (torch.rand(R, generator=g) < P_RATE[rows].repeat_interleave(n)).float()
        all_scores.append(score.view(P, n))
        sh = mb.pack_shard(
            prompts=prompts, prompt_mask=pmask, responses=resp, response_mask=rmask,
            ref_log_probs=torch.randn(R, Lr, generator=g) if with_ref else None,
            rollout_log_probs=torch.randn(R, Lr, generator=g), scores=score,
            entropy_sum=torch.rand(R, generator=g) if with_ref else None, n=n,
            prompt_rows=np.array(rows), dapo_index=np.array(rows) + 1000, data_source=np.array(["d"] * P),
            sample_offset=offset,
        )
        assert sh["sample_index"].min() == offset and sh["sample_index"].max() == offset + n - 1
        mb.save_shard(d, c, sh, Lr)
    mb.write_manifest(d, n_chunks, CHUNK, Lr, [0.1])
    return torch.cat(all_scores)


def main():
    root = tempfile.mkdtemp()
    base, ext = os.path.join(root, "base"), os.path.join(root, "ext")
    s_base = build(base, 4, 0, True, 0)
    s_ext = build(ext, 12, 4, False, 1)
    man = mb.load_manifest(ext)
    assert man["sample_offset"] == 4 and man["has_ref_log_probs"] is False
    try:
        mb.gather_rows(ext, [(0, [0])])
        raise SystemExit("gather_rows must refuse a bank without reference log-probs")
    except AssertionError as e:
        print("baseline-only bank refused for training:", str(e)[:60], "...")
    mb.gather_rows(base, [(0, [0, 1])])  # base bank still loads

    out = os.path.join(base, "baselines", "n16.npz")
    taus = [1.0, 0.1, 0.03]
    subprocess.run(
        [sys.executable, os.path.join(HERE, "compute_master_baselines.py"), "--master", base, "--master", ext,
         "--out", out, "--taus", *map(str, taus), "--check-overlap", "10"],
        check=True,
    )
    summary = json.load(open(out[:-4] + ".json"))
    assert summary["meta"]["n_total"] == 16 and summary["stats"]["overlap"]["cross_bank_duplicates"] == 0

    pooled = torch.cat([s_base, s_ext], 1).double().numpy()  # (P, 16)
    rows = np.array([3, 0, 9, 4, 3, 7])
    b, rbar, info = lookup_master_baseline(out, rows, "mean", 0.1, master_dir=base)
    assert np.allclose(b, pooled.mean(1)[rows]) and np.allclose(rbar, b) and info["n_samples"] == 16
    uid = np.repeat(np.arange(P_TOTAL), 16).astype(str)
    for t in taus:
        b, rbar, _ = lookup_master_baseline(out, rows, "tau_logz", t, master_dir=ext)
        ref = soft_value_baseline(pooled.reshape(-1), uid, t).reshape(P_TOTAL, 16)[:, 0]
        assert np.allclose(b, ref[rows], atol=1e-9), (t, b, ref[rows])
        assert np.allclose(rbar, pooled.mean(1)[rows])
    assert lookup_master_baseline(out, [0], "tau_logz", 0.03)[0][0] == 0.0  # never-solved prompt: b = 0
    assert lookup_master_baseline(out, [9], "tau_logz", 0.03)[0][0] == 1.0  # always solved: b = 1
    for bad in (lambda: lookup_master_baseline(out, rows, "tau_logz", 0.3),
                lambda: lookup_master_baseline(out, rows, "mean", 0.1, master_dir=root)):
        try:
            bad()
            raise SystemExit("expected an assertion")
        except AssertionError as e:
            print("refused:", str(e)[:70])
    # an extension with the base bank's seed or an overlapping sample range is refused
    bad_ext = os.path.join(root, "bad")
    build(bad_ext, 12, 2, False, 0)
    r = subprocess.run([sys.executable, os.path.join(HERE, "compute_master_baselines.py"), "--master", base,
                        "--master", bad_ext, "--out", os.path.join(root, "x.npz")], capture_output=True, text=True)
    assert r.returncode != 0 and "overlap" in r.stderr, r.stderr[-300:]
    print("baselines OK")


if __name__ == "__main__":
    main()
