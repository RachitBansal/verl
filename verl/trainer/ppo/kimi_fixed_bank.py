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
"""Helpers for the fixed-dataset (mu = anchor = pi_ref) Kimi-K2 study: the population optimum of the
fixed-anchor squared regression with a binary reward, and the summary of a rollout bank.

Population optimum (Phase2a_Fixed_PiRef_Dataset_Plan.md §1).  For one prompt with reference pass rate p,
baseline b (= p for the group-mean baseline) and u = pi/pi_ref, minimising
E_{y~pi_ref}(r - b - tau log u)^2 over normalised pi gives u constant within the correct / wrong class with

    tau log u_+ = (1 - b) - lam u_+,   tau log u_- = -b - lam u_-,   p u_+ + (1 - p) u_- = 1 ,

the optimal success rate p* = p u_+, the optimal loss L* = lam^2 (p u_+^2 + (1-p) u_-^2) and the loss at
theta_ref L0 = p (1-b)^2 + (1-p) b^2.  ``solve_floor`` = L*/L0 is the lowest solve ratio a *generalising*
policy can reach on fresh pi_ref samples of that prompt.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np


def _solve_u(tau: float, c: float, lam: float) -> float:
    """u > 0 with tau log u = c - lam u (monotone in log u; bisection)."""
    lo, hi = -300.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tau * mid - (c - lam * math.exp(mid)) > 0:
            hi = mid
        else:
            lo = mid
    return math.exp(0.5 * (lo + hi))


def population_optimum(tau: float, p: float, b: float | None = None) -> dict[str, float]:
    """Per-prompt population optimum of the fixed-anchor K2 regression with binary reward; see module doc."""
    b = p if b is None else b
    if p <= 0.0 or p >= 1.0:
        # zero-advantage prompt: target 0 everywhere, optimum = pi_ref, loss 0
        return {"lam": 0.0, "u_plus": 1.0, "u_minus": 1.0, "p_star": p, "loss_star": 0.0, "loss0": 0.0, "floor": 0.0}

    def gap(lam: float) -> float:
        return p * _solve_u(tau, 1.0 - b, lam) + (1.0 - p) * _solve_u(tau, -b, lam) - 1.0

    lo, hi = 1e-12, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    up, um = _solve_u(tau, 1.0 - b, lam), _solve_u(tau, -b, lam)
    loss0 = p * (1.0 - b) ** 2 + (1.0 - p) * b**2
    loss_star = lam**2 * (p * up**2 + (1.0 - p) * um**2)
    return {
        "lam": lam,
        "u_plus": up,
        "u_minus": um,
        "p_star": p * up,
        "loss_star": loss_star,
        "loss0": loss0,
        "floor": loss_star / loss0,
    }


def bank_floor_summary(pass_counts: np.ndarray, n: int, taus: list[float]) -> dict[str, float]:
    """Dataset-level references from the per-prompt pass counts k (0..n): the population loss floor
    (sum over prompts of L*) / (sum of L0), the optimal reward p* averaged over prompts, and the mean
    optimal shift of correct rows, per tau."""
    out: dict[str, float] = {}
    k = np.asarray(pass_counts, dtype=np.int64)
    counts = np.bincount(k, minlength=n + 1)
    for tau in taus:
        num = den = pstar = logup = n_mixed = 0.0
        for kk, c in enumerate(counts):
            if c == 0:
                continue
            p = kk / n
            opt = population_optimum(tau, p)
            num += c * opt["loss_star"]
            den += c * opt["loss0"]
            pstar += c * opt["p_star"]
            if 0 < kk < n:
                logup += c * math.log(opt["u_plus"])
                n_mixed += c
        t = f"{tau:g}"
        out[f"bank/floor_tau{t}"] = num / max(den, 1e-12)
        out[f"bank/pstar_tau{t}"] = pstar / max(counts.sum(), 1)
        out[f"bank/log_uplus_mixed_tau{t}"] = logup / max(n_mixed, 1)
    return out


def soft_value_baseline(seq_reward: np.ndarray, uid: np.ndarray, tau: float) -> np.ndarray:
    """Per-row baseline b(x) = tau * log( (1/n) sum_j exp(r_j / tau) ) over the row's prompt group (same uid).

    This is the plug-in estimate of tau log Z(x), Z(x) = E_{y~pi_ref} exp(r(x,y)/tau), from the group's own
    pi_ref samples (the A*-PO / OAPL V* estimate).  With it the regression target (r - b)/tau is the log-ratio
    of the KL-regularised optimum pi* = pi_ref exp(r/tau)/Z restricted to the empirical sample, which is a
    normalised distribution, so the targets are feasible: for a binary reward with k of n correct the target is
    -log(k/n + (1 - k/n) e^{-1/tau}) ~ log(n/k) on correct rows and that minus 1/tau on wrong rows.  The
    group-mean baseline instead asks correct rows for (1 - k/n)/tau, which no normalised policy can deliver on
    fresh samples (E_C[Delta] <= log(1/p)).  Groups with a single reward value (all correct / all wrong) get
    b = r exactly (zero target), as with the mean baseline.  Computed in float64 with a stable logsumexp.
    """
    r = np.asarray(seq_reward, dtype=np.float64)
    _, inv = np.unique(np.asarray(uid, dtype=object).astype(str), return_inverse=True)
    n_g = inv.max() + 1
    gmax = np.full(n_g, -np.inf)
    np.maximum.at(gmax, inv, r)
    gmin = np.full(n_g, np.inf)
    np.minimum.at(gmin, inv, r)
    s = np.zeros(n_g)
    np.add.at(s, inv, np.exp((r - gmax[inv]) / tau))
    cnt = np.bincount(inv, minlength=n_g).astype(np.float64)
    b_g = gmax + tau * np.log(s / cnt)
    b_g = np.where(gmax == gmin, gmax, b_g)  # constant-reward group: exact zero advantage
    return b_g[inv]


def lookup_master_baseline(
    path: str, prompt_rows, baseline: str, tau: float, master_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-row baselines from a precomputed baseline file (examples/kimi_k2/fixed_ref/compute_master_baselines.py).

    The npz holds, per prompt row of the master prompt file, ``correct`` (number of correct samples), ``total``
    (N, the samples pooled over the master bank and its extensions), ``mean`` = correct/total, ``taus`` (T,) and
    ``tau_logz`` (T, P) = tau log((1/N) sum_j exp(r_j/tau)), plus ``meta`` (json: the source banks).  Returns
    (b, rbar, info) for the rows' ``prompt_rows``: b = the chosen baseline (``mean`` or ``tau_logz`` at ``tau``,
    which must be one of the file's taus), rbar = the N-sample pass rate, info = summary numbers.
    ``master_dir``, when given, must be one of the file's source banks (guards against a file of another bank)."""
    assert baseline in ("mean", "tau_logz"), f"unknown baseline {baseline}"
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["meta"]))
    if master_dir is not None:
        srcs = [os.path.realpath(s["master_dir"]) for s in meta["sources"]]
        assert os.path.realpath(master_dir) in srcs, f"baseline file {path} was not built from {master_dir}: {srcs}"
    rows = np.asarray(prompt_rows, dtype=np.int64)
    P = len(d["mean"])
    assert rows.min() >= 0 and rows.max() < P, f"prompt rows outside the baseline file's {P} prompts"
    rbar = d["mean"][rows].astype(np.float64)
    if baseline == "mean":
        b = rbar.copy()
    else:
        taus = d["taus"].astype(np.float64)
        hit = np.flatnonzero(np.isclose(taus, tau, rtol=1e-6, atol=0.0))
        assert len(hit) == 1, f"tau={tau} not precomputed in {path} (taus {taus.tolist()})"
        b = d["tau_logz"][int(hit[0])][rows].astype(np.float64)
    total = d["total"][rows]
    info = {
        "n_samples": int(total.min()),
        "n_samples_max": int(total.max()),
        "prompt_pass_rate_mean": float(np.mean(d["mean"][np.unique(rows)])),
        "prompt_zero_or_all_frac": float(
            np.mean(np.isin(d["mean"][np.unique(rows)], (0.0, 1.0)))
        ),
    }
    return b, rbar, info
