# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Master bank: a large pi_ref sample bank (all prompts x n samples) stored once in compact, sharded form, from
which each fixed-dataset run loads a subset (prompts, samples per prompt, held-out roles).

Layout of ``<master_dir>``::

    build_config.json        generation settings (model, n, temperature, lengths, prompt file); checked by every
                             build job so shards from different settings are never mixed
    shards/shard_{c:05d}.pt  prompts [c*chunk, (c+1)*chunk) of the prompt file x n samples (``pack_shard``)
    shards/shard_{c:05d}.json  per-shard sidecar: row counts, per-prompt pass counts, length/entropy sums
    manifest.json            written once every shard exists: shard list, prompt -> shard map, totals
    summary.json             bank statistics over all prompts (pass-count histogram, floors, ...)

A shard stores only real tokens (no padding): per prompt its token ids once, per row (= one sample) the
response tokens, the trainer-computed reference log-probs (the frozen anchor), the vLLM log-probs, the scalar
reward, the summed reference entropy and the sample index.  ``unpack_rows`` rebuilds the padded tensors of a
selection exactly as the rollout produced them (left-padded prompt, right-padded response, verl position ids).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

FORMAT_VERSION = 1


def shard_path(master_dir: str, c: int) -> str:
    return os.path.join(master_dir, "shards", f"shard_{c:05d}.pt")


def atomic_write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def _flat_prefix(x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Concatenate the first lens[i] entries of every row of x (rows of a right-padded tensor)."""
    col = torch.arange(x.shape[1])
    return x[col.unsqueeze(0) < lens.unsqueeze(1)]


def _offsets(lens: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens.to(torch.int64), 0)])


def pack_shard(
    *,
    prompts: torch.Tensor,
    prompt_mask: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor],
    rollout_log_probs: Optional[torch.Tensor],
    scores: torch.Tensor,
    entropy_sum: Optional[torch.Tensor],
    n: int,
    prompt_rows: np.ndarray,
    dapo_index: np.ndarray,
    data_source: np.ndarray,
    extras: Optional[dict] = None,
    sample_offset: int = 0,
) -> dict:
    """Compact form of R = P*n rows that are ``n`` interleaved samples of each of P prompts (row r is sample
    r % n of prompt r // n, as ``DataProto.repeat(interleave=True)`` lays them out).  ``prompts``/``prompt_mask``
    are left-padded, ``responses``/``response_mask`` right-padded with a prefix mask.  ``ref_log_probs`` /
    ``entropy_sum`` may be None (a bank built without the trainer's reference pass, e.g. an extension used only
    for baselines); such a bank cannot be gathered for training.  ``sample_offset``: global number of this bank's
    first sample (an extension of a master bank with n0 samples per prompt uses n0), stored in ``sample_index``."""
    R = responses.shape[0]
    assert R % n == 0, f"{R} rows not a multiple of n={n}"
    P = R // n
    assert len(prompt_rows) == P and len(dapo_index) == P and len(data_source) == P
    pm = prompt_mask.bool().cpu()
    rm = response_mask.bool().cpu()
    rlen = rm.sum(-1)
    # response mask must be a prefix (ones then zeros) for the flat layout to be exact
    assert torch.equal(rm, torch.arange(rm.shape[1]).unsqueeze(0) < rlen.unsqueeze(1)), "response mask not a prefix"
    # every group of n rows shares one prompt
    g = prompts.cpu().view(P, n, -1)
    assert torch.equal(g, g[:, :1].expand_as(g)), "rows of a prompt group have different prompt tokens"
    pm_g = pm.view(P, n, -1)[:, 0]
    plen = pm_g.sum(-1)
    p0 = g[:, 0]
    # left-padded prompt: valid tokens are the last plen entries; flip to a prefix, flatten, keep order
    prompt_tokens = torch.cat([p0[i, -int(plen[i]) :] if plen[i] > 0 else p0[i, :0] for i in range(P)])
    out = {
        "format_version": FORMAT_VERSION,
        "n": int(n),
        "num_prompts": int(P),
        "prompt_rows": torch.as_tensor(np.asarray(prompt_rows, dtype=np.int64)),
        "dapo_index": torch.as_tensor(np.asarray(dapo_index, dtype=np.int64)),
        "data_source": [str(s) for s in data_source],
        "prompt_tokens": prompt_tokens.to(torch.int32),
        "prompt_offsets": _offsets(plen),
        "response_tokens": _flat_prefix(responses.cpu(), rlen).to(torch.int32),
        "response_offsets": _offsets(rlen),
        "ref_log_probs": None if ref_log_probs is None else _flat_prefix(ref_log_probs.cpu().float(), rlen),
        "rollout_log_probs": None if rollout_log_probs is None else _flat_prefix(rollout_log_probs.cpu().float(), rlen),
        "score": scores.cpu().float(),
        "ref_entropy_sum": None if entropy_sum is None else entropy_sum.cpu().float(),
        "sample_offset": int(sample_offset),
        "sample_index": (int(sample_offset) + torch.arange(R) % n).to(torch.int32),
        "extras": {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in (extras or {}).items()},
    }
    return out


def shard_sidecar(sh: dict, max_response_length: int) -> dict:
    """Small json summary of a packed shard (lets the manifest/summary be built without loading shards)."""
    n, P = sh["n"], sh["num_prompts"]
    rlen = (sh["response_offsets"][1:] - sh["response_offsets"][:-1]).double()
    score = sh["score"].double()
    pass_counts = torch.round(score.view(P, n).sum(-1)).to(torch.int64)
    return {
        "format_version": FORMAT_VERSION,
        "n": n,
        "num_prompts": P,
        "num_rows": P * n,
        "prompt_row_start": int(sh["prompt_rows"][0]),
        "prompt_row_end": int(sh["prompt_rows"][-1]) + 1,
        "pass_counts": pass_counts.tolist(),
        "reward_sum": float(score.sum()),
        "response_tokens": int(rlen.sum()),
        "prompt_tokens": int(sh["prompt_offsets"][-1]),
        "clipped_rows": int((rlen >= max_response_length).sum()),
        "ref_entropy_sum": None if sh["ref_entropy_sum"] is None else float(sh["ref_entropy_sum"].double().sum()),
        "sample_offset": int(sh.get("sample_offset", 0)),
        "dapo_index": [int(i) for i in sh["dapo_index"]],
    }


def save_shard(master_dir: str, c: int, sh: dict, max_response_length: int) -> dict:
    """Write shard c and its sidecar atomically (tmp file + rename; the .pt goes last-but-one so a shard is
    considered done only when its sidecar exists)."""
    os.makedirs(os.path.join(master_dir, "shards"), exist_ok=True)
    path = shard_path(master_dir, c)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(sh, tmp)
    os.replace(tmp, path)
    side = shard_sidecar(sh, max_response_length)
    atomic_write_json(path[:-3] + ".json", side)
    return side


def shard_done(master_dir: str, c: int) -> bool:
    p = shard_path(master_dir, c)
    return os.path.exists(p) and os.path.exists(p[:-3] + ".json")


# ---- alternative anchors: the same rows' reference log-probs recomputed under another numerical precision -------
# anchors/<tag>/shard_XXXXX.pt holds one flat fp32 tensor in the shard's response_tokens layout (row-major, real
# tokens only); anchors/<tag>/shard_XXXXX.json is its sidecar (written last = done marker).
def anchor_path(master_dir: str, tag: str, c: int) -> str:
    return os.path.join(master_dir, "anchors", tag, f"shard_{c:05d}.pt")


def save_anchor(master_dir: str, tag: str, c: int, flat: torch.Tensor, stats: dict) -> None:
    path = anchor_path(master_dir, tag, c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(flat.float().contiguous().cpu(), tmp)
    os.replace(tmp, path)
    atomic_write_json(path[:-3] + ".json", stats)


def anchor_done(master_dir: str, tag: str, c: int) -> bool:
    p = anchor_path(master_dir, tag, c)
    return os.path.exists(p) and os.path.exists(p[:-3] + ".json")


def write_manifest(master_dir: str, num_chunks: int, chunk_prompts: int, max_response_length: int, taus) -> dict:
    """Combine the sidecars into manifest.json and summary.json (call once every shard is done)."""
    from verl.trainer.ppo.kimi_fixed_bank import bank_floor_summary

    sides = []
    for c in range(num_chunks):
        with open(shard_path(master_dir, c)[:-3] + ".json") as f:
            sides.append(json.load(f))
    n = sides[0]["n"]
    assert all(s["n"] == n for s in sides)
    offset = int(sides[0].get("sample_offset", 0))
    assert all(int(s.get("sample_offset", 0)) == offset for s in sides)
    has_ref = all(s["ref_entropy_sum"] is not None for s in sides)
    pass_counts = np.concatenate([np.asarray(s["pass_counts"], dtype=np.int64) for s in sides])
    rows = sum(s["num_rows"] for s in sides)
    resp_tok = sum(s["response_tokens"] for s in sides)
    manifest = {
        "format_version": FORMAT_VERSION,
        "n": n,
        "sample_offset": offset,
        "has_ref_log_probs": has_ref,
        "chunk_prompts": chunk_prompts,
        "num_shards": num_chunks,
        "num_prompts": int(len(pass_counts)),
        "num_rows": int(rows),
        "shards": [
            {
                "file": os.path.basename(shard_path(master_dir, c)),
                "prompt_row_start": s["prompt_row_start"],
                "prompt_row_end": s["prompt_row_end"],
                "num_prompts": s["num_prompts"],
            }
            for c, s in enumerate(sides)
        ],
    }
    summary = {
        "master/prompts": int(len(pass_counts)),
        "master/rows": int(rows),
        "master/n": n,
        "master/reward_mean": sum(s["reward_sum"] for s in sides) / rows,
        "master/response_length_mean": resp_tok / rows,
        "master/prompt_length_mean": sum(s["prompt_tokens"] for s in sides) / len(pass_counts),
        "master/clip_ratio": sum(s["clipped_rows"] for s in sides) / rows,
        "master/entropy_ref_token_mean": sum(s["ref_entropy_sum"] for s in sides) / resp_tok if has_ref else None,
        "master/pass_at_n": float((pass_counts > 0).mean()),
        "master/zero_adv_prompt_frac": float(((pass_counts == 0) | (pass_counts == n)).mean()),
    }
    for kk, cnt in enumerate(np.bincount(pass_counts, minlength=n + 1)):
        summary[f"master/pass_count_hist/{kk}"] = int(cnt)
    summary.update({k.replace("bank/", "master/"): v for k, v in bank_floor_summary(pass_counts, n, taus).items()})
    atomic_write_json(os.path.join(master_dir, "summary.json"), summary)
    atomic_write_json(os.path.join(master_dir, "manifest.json"), manifest)
    return summary


def load_manifest(master_dir: str) -> dict:
    with open(os.path.join(master_dir, "manifest.json")) as f:
        return json.load(f)


def gather_rows(master_dir: str, requests: list[tuple[int, list[int]]], anchor: Optional[str] = None) -> dict:
    """Load the compact pieces of the requested rows.  ``requests`` is a list of (prompt_row, sample indices);
    the output rows are in request order (all samples of a request contiguous).  Loads each needed shard once.
    ``anchor``: tag of an alternative anchor (anchors/<tag>/) whose log-probs replace the stored
    ``ref_log_probs``."""
    man = load_manifest(master_dir)
    n = man["n"]
    assert man.get("has_ref_log_probs", True), (
        f"master bank {master_dir} was built without reference log-probs (baseline-only extension); "
        "it cannot be loaded for training"
    )
    starts = np.array([s["prompt_row_start"] for s in man["shards"]])
    by_shard: dict[int, list[int]] = {}
    for i, (prow, samples) in enumerate(requests):
        c = int(np.searchsorted(starts, prow, side="right") - 1)
        assert 0 <= c and prow < man["shards"][c]["prompt_row_end"], f"prompt row {prow} not in the master bank"
        assert all(0 <= s < n for s in samples), f"sample index out of range 0..{n - 1}"
        by_shard.setdefault(c, []).append(i)
    pieces: list[Optional[dict]] = [None] * len(requests)
    for c, req_ids in sorted(by_shard.items()):
        sh = torch.load(os.path.join(master_dir, "shards", man["shards"][c]["file"]), weights_only=False)
        if anchor:
            assert anchor_done(master_dir, anchor, c), f"anchor {anchor}: shard {c} missing (MODE=master_anchor)"
            alt = torch.load(anchor_path(master_dir, anchor, c))
            assert alt.shape == sh["ref_log_probs"].shape, f"anchor {anchor} shard {c}: shape mismatch"
            sh["ref_log_probs"] = alt
        p0 = int(sh["prompt_rows"][0])
        po, ro = sh["prompt_offsets"], sh["response_offsets"]
        for i in req_ids:
            prow, samples = requests[i]
            lp = prow - p0
            assert int(sh["prompt_rows"][lp]) == prow
            rows = [lp * n + s for s in samples]
            pieces[i] = {
                "prompt_row": prow,
                "dapo_index": int(sh["dapo_index"][lp]),
                "data_source": sh["data_source"][lp],
                "prompt": sh["prompt_tokens"][po[lp] : po[lp + 1]].clone(),
                "samples": list(samples),
                "responses": [sh["response_tokens"][ro[r] : ro[r + 1]].clone() for r in rows],
                "ref_log_probs": [sh["ref_log_probs"][ro[r] : ro[r + 1]].clone() for r in rows],
                "rollout_log_probs": None
                if sh["rollout_log_probs"] is None
                else [sh["rollout_log_probs"][ro[r] : ro[r + 1]].clone() for r in rows],
                "score": sh["score"][rows].clone(),
                "ref_entropy_sum": sh["ref_entropy_sum"][rows].clone(),
            }
        del sh
    return {"n": n, "pieces": pieces}


def unpack_rows(pieces: list[dict], pad_token_id: int, max_prompt_length: int, max_response_length: int) -> dict:
    """Padded tensors for the rows of ``pieces`` (one row per (piece, sample), piece order), matching the vLLM
    rollout's layout: prompts left-padded to max_prompt_length, responses right-padded to max_response_length,
    attention mask = [prompt mask | response prefix mask], position ids = clip(cumsum(prompt mask) - 1, 0) on
    the prompt and last prompt position + 1, 2, ... on the response, token-level score at the last response
    token, reference log-probs as old_log_probs (0 beyond the response), vLLM log-probs (-1 beyond)."""
    resp = [r for p in pieces for r in p["responses"]]
    R = len(resp)
    Lp, Lr = max_prompt_length, max_response_length
    prompts = torch.full((R, Lp), pad_token_id, dtype=torch.int64)
    pmask = torch.zeros((R, Lp), dtype=torch.int64)
    responses = torch.full((R, Lr), pad_token_id, dtype=torch.int64)
    rmask = torch.zeros((R, Lr), dtype=torch.int64)
    old_lp = torch.zeros((R, Lr), dtype=torch.float32)
    has_roll = pieces[0]["rollout_log_probs"] is not None
    roll_lp = torch.full((R, Lr), -1.0, dtype=torch.float32) if has_roll else None
    scores = torch.zeros((R, Lr), dtype=torch.float32)
    r = 0
    for p in pieces:
        pt = p["prompt"].to(torch.int64)
        assert len(pt) <= Lp, f"prompt of {len(pt)} tokens exceeds max_prompt_length {Lp}"
        for j in range(len(p["responses"])):
            if len(pt):
                prompts[r, Lp - len(pt) :] = pt
                pmask[r, Lp - len(pt) :] = 1
            rt = p["responses"][j].to(torch.int64)
            L = len(rt)
            assert L <= Lr, f"response of {L} tokens exceeds max_response_length {Lr}"
            responses[r, :L] = rt
            rmask[r, :L] = 1
            old_lp[r, :L] = p["ref_log_probs"][j]
            if has_roll:
                roll_lp[r, :L] = p["rollout_log_probs"][j]
            scores[r, max(L - 1, 0)] = p["score"][j]
            r += 1
    ppos = torch.clip(torch.cumsum(pmask, dim=-1) - 1, min=0)
    rpos = ppos[:, -1:] + torch.arange(1, Lr + 1).unsqueeze(0)
    out = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": torch.cat([prompts, responses], dim=-1),
        "attention_mask": torch.cat([pmask, rmask], dim=-1),
        "position_ids": torch.cat([ppos, rpos], dim=-1),
        "response_mask": rmask,
        "old_log_probs": old_lp,
        "ref_log_probs": old_lp.clone(),
        "token_level_scores": scores,
        "token_level_rewards": scores.clone(),
    }
    if has_roll:
        out["rollout_log_probs"] = roll_lp
    return out


# ---- per-prompt baselines from all samples of one or more banks over the same prompt file -----------------------
# compute_master_baselines.py combines a master bank and its extensions (same prompts, disjoint sample ranges) into
# <out>.npz with, per prompt row of the prompt file: the number of samples N, the number correct, the mean
# baseline (= pass rate) and tau log Zhat for a list of tau.  A run selects one column with
# trainer.fixed_dataset.baseline_file + baseline=mean|tau_logz (ray_trainer._rebaseline_fixed_bank).
BASELINE_KEYS_CHECKED = ("model", "prompt_file", "num_prompts", "temperature", "top_p", "top_k", "max_prompt_length",
                         "max_response_length")


def load_pass_counts(master_dir: str) -> dict:
    """Per-prompt pass counts of a complete bank from its sidecars (no shard loads): {"n", "sample_offset",
    "pass_counts" (num_prompts,), "reward_sum", "dapo_index" (or None for sidecars written before it was stored),
    "build_config"}.  Requires binary rewards: the sidecar's reward_sum must equal its summed pass counts."""
    man = load_manifest(master_dir)
    with open(os.path.join(master_dir, "build_config.json")) as f:
        cfg = json.load(f)
    counts, dapo, reward_sum = [], [], 0.0
    expect = 0
    for s_ in man["shards"]:
        with open(os.path.join(master_dir, "shards", s_["file"][:-3] + ".json")) as f:
            side = json.load(f)
        assert side["prompt_row_start"] == expect, f"{master_dir}: shard rows not contiguous at {expect}"
        expect = side["prompt_row_end"]
        pc = np.asarray(side["pass_counts"], dtype=np.int64)
        assert abs(float(pc.sum()) - float(side["reward_sum"])) < 1e-3, (
            f"{master_dir} {s_['file']}: reward_sum {side['reward_sum']} != pass-count sum {pc.sum()} "
            "(non-binary reward: pass counts do not determine the baselines)"
        )
        counts.append(pc)
        dapo.append(side.get("dapo_index"))
        reward_sum += float(side["reward_sum"])
    has_dapo = all(d is not None for d in dapo)
    return {
        "n": int(man["n"]),
        "sample_offset": int(man.get("sample_offset", 0)),
        "pass_counts": np.concatenate(counts),
        "reward_sum": reward_sum,
        "dapo_index": np.concatenate([np.asarray(d, dtype=np.int64) for d in dapo]) if has_dapo else None,
        "build_config": cfg,
    }


def tau_logz_from_counts(correct: np.ndarray, total: np.ndarray, tau: float) -> np.ndarray:
    """tau log Zhat, Zhat = (1/N) sum_j exp(r_j/tau), for binary rewards with ``correct`` of ``total`` samples:
    tau * log((N-k)/N + (k/N) e^{1/tau}) in float64 (stable logaddexp); exactly 0 for k = 0 and 1 for k = N
    (the same values kimi_fixed_bank.soft_value_baseline gives a constant group)."""
    k = np.asarray(correct, dtype=np.float64)
    N = np.asarray(total, dtype=np.float64)
    p = k / N
    with np.errstate(divide="ignore"):
        b = tau * np.logaddexp(np.log1p(-p), 1.0 / tau + np.log(p))
    b = np.where(k == 0, 0.0, b)
    b = np.where(k == N, 1.0, b)
    return b


# ---- sub-sampled training (trainer.fixed_dataset.rotate_k) ---------------------------------------------------------
def rotation_samples(pools: list, k: int, step: int, seed: int = 0) -> list:
    """Samples of each prompt trained on at ``step`` (1-based) when every step takes ``k`` samples per prompt from
    its pool (a sorted array of master sample indices).  Within an epoch of len(pool) // k steps every pool sample
    is used exactly once; the order is reshuffled every epoch (rng keyed by seed, epoch and pool size, so it is a
    pure function of the step: a resumed run continues the same schedule).  k >= len(pool): the whole pool every
    step (full batch).  Returns one sorted int64 array per prompt."""
    assert step >= 1 and k >= 1
    out: list = [None] * len(pools)
    by_size: dict[int, list[int]] = {}
    for i, p in enumerate(pools):
        by_size.setdefault(len(p), []).append(i)
    for size, idx in by_size.items():
        arr = np.stack([np.asarray(pools[i], dtype=np.int64) for i in idx])  # (P, size)
        if k >= size:
            for j, i in enumerate(idx):
                out[i] = arr[j].copy()
            continue
        assert size % k == 0, f"pool of {size} samples is not a multiple of rotate_k={k}"
        per_epoch = size // k
        epoch, slot = divmod(step - 1, per_epoch)
        keys = np.random.default_rng([int(seed), int(epoch), int(size)]).random((len(idx), size))
        pick = np.argsort(keys, axis=1)[:, slot * k : (slot + 1) * k]
        chosen = np.sort(np.take_along_axis(arr, pick, axis=1), axis=1)
        for j, i in enumerate(idx):
            out[i] = chosen[j]
    return out


def subset_piece(piece: dict, samples) -> dict:
    """The rows of ``piece`` (from gather_rows) for the given sample indices (a subset of piece["samples"])."""
    pos = {s: j for j, s in enumerate(piece["samples"])}
    js = [pos[int(s)] for s in samples]
    out = dict(piece)
    out["samples"] = [int(s) for s in samples]
    for key in ("responses", "ref_log_probs", "rollout_log_probs"):
        if piece[key] is not None:
            out[key] = [piece[key][j] for j in js]
    for key in ("score", "ref_entropy_sum"):
        out[key] = piece[key][js]
    return out
