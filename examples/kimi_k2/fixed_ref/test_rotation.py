"""CPU test of sub-sampled training (trainer.fixed_dataset.rotate_k): kimi_master_bank.rotation_samples (each
epoch uses every pool sample once, reshuffled per epoch, pure function of the step, k >= pool = whole pool) and
RayPPOTrainer._rotation_chunks (the step's rows split into update chunks: exactly the chosen samples, A = r - b(x),
probe prompts + tracked rows in the last chunk, chunk sizes <= max_update_rows) and _merge_chunk_metrics.
Run: python test_rotation.py"""

import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../.."))
from verl.trainer.ppo import kimi_master_bank as mb  # noqa: E402
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _merge_chunk_metrics  # noqa: E402

# ---- rotation_samples
pools = [np.arange(8)] * 5 + [np.array([0, 1, 4, 5, 6, 7])] * 3
for k in (1, 2):
    for size, idx in ((8, range(5)), (6, range(5, 8))):
        E = size // k
        for e in range(3):
            got = [mb.rotation_samples(pools, k, e * E + s + 1, seed=3) for s in range(E)]
            for i in idx:
                allc = np.concatenate([g[i] for g in got])
                assert sorted(allc.tolist()) == sorted(pools[i].tolist()), (k, size, e, i, allc)
        a = mb.rotation_samples(pools, k, 1, seed=3)
        b = mb.rotation_samples(pools, k, 1 + 8 // k, seed=3)  # same slot, next epoch (size-8 pools)
        assert any(not np.array_equal(a[i], b[i]) for i in range(5)), "epochs must be reshuffled"
assert all(np.array_equal(x, y) for x, y in zip(mb.rotation_samples(pools, 2, 7, 3), mb.rotation_samples(pools, 2, 7, 3)))
full = mb.rotation_samples(pools, 8, 5)
assert all(np.array_equal(full[i], pools[i]) for i in range(8))
try:
    mb.rotation_samples([np.arange(6)], 4, 1)
    raise SystemExit("pool not a multiple of k must fail")
except AssertionError:
    pass
print("rotation_samples OK")

# ---- a small master bank
g = torch.Generator().manual_seed(0)
PAD, Lp, Lr, P, N, CH = 7, 6, 8, 10, 8, 4
root = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rot_test_{os.getpid()}")
os.makedirs(root, exist_ok=True)
mb.atomic_write_json(os.path.join(root, "build_config.json"), {"n": N})
for c in range((P + CH - 1) // CH):
    rows = list(range(c * CH, min(P, (c + 1) * CH)))
    R = len(rows) * N
    prompts = torch.full((R, Lp), PAD)
    prompts[:, -3:] = torch.tensor(rows).repeat_interleave(N)[:, None] + 100
    pm = torch.zeros((R, Lp), dtype=torch.int64)
    pm[:, -3:] = 1
    rl = torch.randint(1, Lr + 1, (R,), generator=g)
    rm = (torch.arange(Lr)[None] < rl[:, None]).long()
    resp = torch.where(rm.bool(), torch.randint(200, 300, (R, Lr), generator=g), torch.full((R, Lr), PAD))
    sh = mb.pack_shard(prompts=prompts, prompt_mask=pm, responses=resp, response_mask=rm,
                       ref_log_probs=torch.randn(R, Lr, generator=g), rollout_log_probs=torch.randn(R, Lr, generator=g),
                       scores=(torch.rand(R, generator=g) < 0.4).float(), entropy_sum=torch.rand(R, generator=g), n=N,
                       prompt_rows=np.array(rows), dapo_index=np.array(rows), data_source=np.array(["d"] * len(rows)))
    mb.save_shard(root, c, sh, Lr)
mb.write_manifest(root, (P + CH - 1) // CH, CH, Lr, [0.1])

fake = types.SimpleNamespace(
    tokenizer=types.SimpleNamespace(pad_token_id=PAD),
    config=types.SimpleNamespace(data=types.SimpleNamespace(max_prompt_length=Lp, max_response_length=Lr)),
)
fake._master_dataproto = types.MethodType(RayPPOTrainer._master_dataproto, fake)
train = np.arange(8)  # prompts 8, 9: held-out fresh prompts (tracked)
tpools = [np.arange(N) if p >= 2 else np.array([0, 1, 4, 5, 6, 7]) for p in train]  # prompts 0,1: samples 2,3 held out
probe = np.array([False] * 6 + [True] * 2)
pieces = mb.gather_rows(root, [(int(p), tp.tolist()) for p, tp in zip(train, tpools)])["pieces"]
bvals = np.linspace(0.1, 0.8, 8)
tr = fake._master_dataproto(mb.gather_rows(root, [(8, list(range(4))), (9, list(range(4)))])["pieces"],
                            [(1, "holdout_prompt:8", "holdout_prompt:8"), (1, "holdout_prompt:9", "holdout_prompt:9")])
tr.batch["advantages"] = torch.zeros_like(tr.batch["response_mask"], dtype=torch.float32)
tr.batch["returns"] = tr.batch["advantages"].clone()
tr.non_tensor_batch.update(ref_pass_rate=np.zeros(8, np.float32), kimi_probe=np.zeros(8, np.int64),
                           kimi_baseline=np.zeros(8, np.float32))
for k, max_rows in ((2, 0), (2, 5), (8, 12), (8, 0)):
    rot = {"k": k, "seed": 0, "max_rows": max_rows, "pieces": pieces, "pools": tpools, "b": bvals, "rbar": bvals,
           "probe": probe, "uids": np.array([f"train:{p}" for p in train], dtype=object), "n_nonprobe": 6}
    for step in (1, 2, 5):
        chosen = mb.rotation_samples(tpools, k, step, 0)
        chunks = list(RayPPOTrainer._rotation_chunks(fake, rot, step, tr))
        assert sum(c[1] for c in chunks) == chunks[0][2] == sum(len(c) for c in chosen)
        assert [c[3] for c in chunks] == [False] * (len(chunks) - 1) + [True]
        seen = {}
        for rows, n_up, total, last in chunks:
            nt = rows.non_tensor_batch
            fl = nt["kimi_holdout"]
            assert (fl == 1).sum() == (8 if last else 0) and (fl == 0).sum() == n_up
            if max_rows and not last:
                assert n_up <= max_rows, (n_up, max_rows)
            if not last:
                assert nt["kimi_probe"].sum() == 0
            r = rows.batch["token_level_scores"].sum(-1).numpy()
            pr = nt["master_prompt_row"]
            m = fl == 0
            assert np.allclose(rows.batch["advantages"].sum(-1).numpy()[m] / rows.batch["response_mask"].sum(-1).numpy()[m],
                               r[m] - bvals[pr[m]], atol=1e-6)
            assert np.allclose(nt["kimi_baseline"][m], bvals[pr[m]])
            for p_, s_ in zip(pr[m], nt["master_sample"][m]):
                seen.setdefault(int(p_), []).append(int(s_))
        for i, p in enumerate(train):
            assert sorted(seen[int(p)]) == chosen[i].tolist(), (k, step, p, seen[int(p)], chosen[i])
        assert int(chunks[-1][0].non_tensor_batch["kimi_probe"].sum()) == sum(len(chosen[i]) for i in (6, 7))
    print(f"_rotation_chunks k={k} max_update_rows={max_rows}: {len(chunks)} chunk(s) OK")

m = _merge_chunk_metrics([(10, {"actor/loss_exact": 0.25, "a": 1.0, "x_max": 3}),
                          (30, {"actor/loss_exact": 0.5, "a": 2.0, "x_max": 5, "actor/grad_norm": 7.0})],
                         sum_keys=("actor/loss_exact",))
assert m == {"actor/loss_exact": 0.75, "a": 1.75, "x_max": 5, "actor/grad_norm": 7.0}, m
print("rotation OK")
