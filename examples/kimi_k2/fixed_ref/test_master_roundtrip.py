"""CPU roundtrip test of the master-bank shard format: pack -> save -> manifest -> gather -> unpack equals the
rollout tensors (prompts, responses, masks, position ids, log-probs, score).  Run: python test_master_roundtrip.py"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from verl.trainer.ppo import kimi_master_bank as mb  # noqa: E402
from verl.utils.model import compute_position_id_with_mask  # noqa: E402

g = torch.Generator().manual_seed(0)
PAD, Lp, Lr, n = 7, 12, 10, 4
KEYS = ["prompts", "responses", "attention_mask", "position_ids", "response_mask", "old_log_probs", "rollout_log_probs"]


def make(P: int, row0: int):
    """P prompts x n samples laid out like the vLLM rollout (left-padded prompts, right-padded responses)."""
    R = P * n
    plen = torch.randint(1, Lp + 1, (P,), generator=g)
    prompts = torch.full((P, Lp), PAD)
    pmask = torch.zeros((P, Lp), dtype=torch.int64)
    for i in range(P):
        prompts[i, Lp - plen[i] :] = torch.randint(100, 200, (int(plen[i]),), generator=g)
        pmask[i, Lp - plen[i] :] = 1
    prompts = prompts.repeat_interleave(n, 0)
    pmask = pmask.repeat_interleave(n, 0)
    rlen = torch.randint(1, Lr + 1, (R,), generator=g)
    rmask = (torch.arange(Lr)[None] < rlen[:, None]).long()
    resp = torch.where(rmask.bool(), torch.randint(200, 300, (R, Lr), generator=g), torch.full((R, Lr), PAD))
    lp = torch.randn(R, Lr, generator=g)
    rlp = torch.randn(R, Lr, generator=g)
    score = (torch.rand(R, generator=g) < 0.4).float()
    ent = torch.rand(R, Lr, generator=g)
    ppos = compute_position_id_with_mask(pmask)
    pos = torch.cat([ppos, ppos[:, -1:] + torch.arange(1, Lr + 1)[None]], -1)
    ref = {
        "prompts": prompts,
        "responses": resp,
        "attention_mask": torch.cat([pmask, rmask], -1),
        "position_ids": pos,
        "response_mask": rmask,
        "old_log_probs": lp * rmask,
        "rollout_log_probs": torch.where(rmask.bool(), rlp, torch.tensor(-1.0)),
        "score": score,
    }
    sh = mb.pack_shard(
        prompts=prompts,
        prompt_mask=pmask,
        responses=resp,
        response_mask=rmask,
        ref_log_probs=lp,
        rollout_log_probs=rlp,
        scores=score,
        entropy_sum=(ent * rmask).sum(-1),
        n=n,
        prompt_rows=np.arange(row0, row0 + P),
        dapo_index=np.arange(row0, row0 + P) + 1000,
        data_source=np.array(["math_dapo"] * P),
    )
    return sh, ref


def main():
    d = tempfile.mkdtemp()
    layout = [(3, 0), (3, 3), (2, 6)]  # (prompts, first prompt row) per shard
    refs = {}
    for c, (P, row0) in enumerate(layout):
        sh, ref = make(P, row0)
        mb.save_shard(d, c, sh, Lr)
        refs[c] = (row0, ref)
    summ = mb.write_manifest(d, len(layout), 3, Lr, [0.1])
    print(json.dumps({k: v for k, v in summ.items() if "hist" not in k}))
    reqs = [(4, [1, 3]), (0, [0, 1]), (7, [2, 0])]
    out = mb.unpack_rows(mb.gather_rows(d, reqs)["pieces"], PAD, Lp, Lr)
    i = 0
    for prow, samples in reqs:
        c = max(c for c, (row0, _) in refs.items() if row0 <= prow)
        row0, R = refs[c]
        for s in samples:
            r = (prow - row0) * n + s
            for key in KEYS:
                assert torch.equal(out[key][i], R[key][r].to(out[key].dtype)), (key, prow, s)
            L = int(R["response_mask"][r].sum())
            assert out["token_level_scores"][i].sum() == R["score"][r]
            assert out["token_level_scores"][i, L - 1] == R["score"][r]
            assert torch.equal(out["input_ids"][i], torch.cat([R["prompts"][r], R["responses"][r]]))
            i += 1
    print("roundtrip OK", i, "rows")

    # alternative anchor: per-shard flat tensors in the response layout replace ref_log_probs (old_log_probs)
    for c, (row0, R) in refs.items():
        m = R["response_mask"].bool()
        mb.save_anchor(d, "alt", c, (R["old_log_probs"] + 1000.0)[m], {"c": c})
    out2 = mb.unpack_rows(mb.gather_rows(d, reqs, anchor="alt")["pieces"], PAD, Lp, Lr)
    m = out2["response_mask"].bool()
    assert torch.equal(out2["old_log_probs"][m], out["old_log_probs"][m] + 1000.0)
    assert torch.equal(out2["old_log_probs"][~m], out["old_log_probs"][~m])
    for key in ("responses", "attention_mask", "position_ids", "rollout_log_probs"):
        assert torch.equal(out2[key], out[key]), key
    print("anchor substitution OK")


if __name__ == "__main__":
    main()
