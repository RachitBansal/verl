"""Data layer for the ICLR figures: wandb trajectories -> per-run crossings -> per-cell best LR.

Input: long CSV from fetch_all.py (run,id,state,created,step,aime_mean1,resp_len,prompt_len,...)
       + all_runs.jsonl (per-id config: kl coef etc.).

Protocol (matches Clara's plots/scripts/pull_steps_to_50*.py):
  * metric  = AIME 1983-2024 mean@1 (validation, logged every step)
  * S(tau)  = first crossing of tau, linearly interpolated between the two bracketing readings
  * resume segments (same run name, or name + "_resume") are stitched into one trajectory
  * per cell (setting, batch axis value): best run = fastest S(tau) over all LR x KL tried
"""
from __future__ import annotations
import csv, json, re
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

METRIC_TAU = 0.5


def parse_name(name: str) -> dict:
    d = dict(kind="std", K=None, B=None, dsk=None, N=None, lr=None, seed="", upd="updated_scaling" in name,
             kl_name=None)
    m = re.search(r"seed(\d+)", name); d["seed"] = m.group(1) if m else ""
    m = re.search(r"_kl([0-9.e-]+?)(?:_|$)", name); d["kl_name"] = m.group(1) if m else None
    if name.startswith("downsample"):
        d["kind"] = "ds"
        m = re.search(r"downsample_n(\d+)_dsk(\d+)_bsz(\d+)_lr([0-9.]+e-?\d+)", name)
        if m:
            d["N"], d["dsk"], d["B"], d["lr"] = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
    else:
        m = re.search(r"^n(\d+)_bsz(\d+)(?:_lr([0-9.]+e-?\d+))?", name)
        if m:
            d["K"], d["B"] = int(m.group(1)), int(m.group(2))
            d["lr"] = float(m.group(3)) if m.group(3) else None
    return d


@dataclass
class Run:
    key: str                # logical run name (resume suffix stripped)
    ids: list
    meta: dict
    kl: float | None
    state: str
    steps: np.ndarray = field(default_factory=lambda: np.array([]))
    vals: np.ndarray = field(default_factory=lambda: np.array([]))
    tok_per_seq: np.ndarray = field(default_factory=lambda: np.array([]))  # prompt+response mean length per step

    @property
    def last_step(self):
        return int(self.steps.max()) if len(self.steps) else 0

    @property
    def max_val(self):
        return float(np.nanmax(self.vals)) if len(self.vals) else float("nan")

    def crossing(self, tau: float, sustained: int = 1):
        """Interpolated first step at which mean@1 >= tau. sustained=k requires k consecutive readings >= tau."""
        s, v = self.steps, self.vals
        for i in range(len(v)):
            if v[i] >= tau and all(v[j] >= tau for j in range(i, min(len(v), i + sustained))):
                if i == 0:
                    return float(s[0])
                if v[i] == v[i - 1]:
                    return float(s[i])
                return float(s[i - 1] + (s[i] - s[i - 1]) * (tau - v[i - 1]) / (v[i] - v[i - 1]))
        return None

    def tokens_per_seq_until(self, step):
        """Mean (prompt + response) tokens per sequence over training steps <= step."""
        if not len(self.tok_per_seq):
            return float("nan")
        m = (self.steps <= step) & np.isfinite(self.tok_per_seq)
        return float(self.tok_per_seq[m].mean()) if m.any() else float("nan")


def load_runs(traj_csv: str, meta_jsonl: str | None = None) -> dict[str, Run]:
    kl_by_id, state_by_id = {}, {}
    if meta_jsonl:
        for line in open(meta_jsonl):
            r = json.loads(line)
            kl_by_id[r["id"]] = r.get("kl"); state_by_id[r["id"]] = r.get("state")
    acc = defaultdict(lambda: defaultdict(list))   # key -> step -> [(val, tok)]
    ids = defaultdict(set); states = defaultdict(set); names = {}
    with open(traj_csv) as f:
        for r in csv.DictReader(f):
            name = r["run"]
            key = re.sub(r"_resume\d*", "", name)
            ids[key].add(r["id"]); states[key].add(r.get("state", "")); names[key] = name
            v = r.get("aime_mean1")
            v = float(v) if v not in (None, "") else None
            rl, pl = r.get("resp_len"), r.get("prompt_len")
            tok = (float(rl) + float(pl)) if rl not in (None, "") and pl not in (None, "") else None
            acc[key][int(r["step"])].append((v, tok))
    runs = {}
    for key, d in acc.items():
        meta = parse_name(key)
        steps, vals, toks = [], [], []
        for s in sorted(d):
            vs = [x[0] for x in d[s] if x[0] is not None]
            ts = [x[1] for x in d[s] if x[1] is not None]
            steps.append(s); vals.append(np.mean(vs) if vs else np.nan); toks.append(np.mean(ts) if ts else np.nan)
        steps, vals, toks = map(np.array, (steps, vals, toks))
        m = np.isfinite(vals)
        kls = {kl_by_id.get(i) for i in ids[key]} - {None}
        kl = kls.pop() if len(kls) == 1 else (float(meta["kl_name"].replace("1e-2", "0.01").replace("1e-3", "0.001")) if meta["kl_name"] else None)
        if kl is None and meta["kind"] == "std":
            kl = 0.001   # default in on_policy.sh
        state = "running" if "running" in states[key] else ("finished" if "finished" in states[key] else "crashed")
        run = Run(key=key, ids=sorted(ids[key]), meta=meta, kl=kl, state=state)
        run.steps, run.vals = steps[m], vals[m]
        # tokens per sequence are logged on training steps (every step); keep the full-resolution series
        tm = np.isfinite(toks)
        run.tok_per_seq = np.interp(run.steps, steps[tm], toks[tm]) if tm.sum() >= 2 else np.full(len(run.steps), np.nan)
        runs[key] = run
    return runs


# ----------------------------------------------------------------------------- cells
def cell_key(run: Run):
    m = run.meta
    if m["kind"] == "ds":
        return ("ds", m["N"], m["B"], m["dsk"])
    return ("std", m["K"], m["B"])


def best_per_cell(runs, tau=METRIC_TAU, select=None, sustained=1, min_steps=0):
    """Return {cell: dict(best=Run or None, S=float|None, tried=[(run, S, lr, kl)], ...)}.
    select(run) -> bool filters runs before grouping."""
    cells = defaultdict(list)
    for r in runs.values():
        if select and not select(r):
            continue
        if r.meta["lr"] is None or (r.meta["kind"] == "std" and r.meta["K"] is None):
            continue
        if len(r.steps) == 0:
            continue
        S = r.crossing(tau, sustained=sustained)
        cells[cell_key(r)].append(dict(run=r, S=S, lr=r.meta["lr"], kl=r.kl, upd=r.meta["upd"],
                                        last=r.last_step, max_val=r.max_val))
    out = {}
    for c, tried in cells.items():
        hit = [t for t in tried if t["S"] is not None]
        best = min(hit, key=lambda t: t["S"]) if hit else None
        out[c] = dict(best=best, tried=sorted(tried, key=lambda t: (t["lr"], t["kl"] or 0)),
                      S=best["S"] if best else None, longest=max(t["last"] for t in tried))
    return out


# ----------------------------------------------------------------------------- fits
def fit_mccandlish(x, S):
    """S(x) = S_min (1 + x*/x); least squares in log S. Returns (S_min, x_star)."""
    x = np.asarray(x, float); S = np.asarray(S, float)
    best = None
    for lx in np.linspace(np.log2(x.min()) - 4, np.log2(x.max()) + 6, 2000):
        xs = 2 ** lx
        logSmin = np.mean(np.log(S) - np.log1p(xs / x))
        res = np.sum((np.log(S) - logSmin - np.log1p(xs / x)) ** 2)
        if best is None or res < best[0]:
            best = (res, np.exp(logSmin), xs)
    return best[1], best[2]


def fit_powerlaw(x, y):
    """y = a x^alpha (log-log OLS). Returns a, alpha, stderr(alpha)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    alpha, loga = coef
    n = len(lx)
    se = float("nan")
    if n > 2:
        resid = ly - A @ coef
        s2 = (resid ** 2).sum() / (n - 2)
        se = float(np.sqrt(s2 / ((lx - lx.mean()) ** 2).sum()))
    return float(np.exp(loga)), float(alpha), se


# ----------------------------------------------------------------------------- matched comparisons
def hparam(run: Run):
    """(learning rate, KL coefficient) of a run."""
    return (run.meta["lr"], run.kl)


def common_hparams(runs, sel_a, sel_b, key_a, key_b):
    """For two arms of a comparison, the (lr, kl) pairs that were run in BOTH arms at each x (e.g. K).
    sel_*: run -> bool; key_*: run -> x. Returns {x: set((lr, kl))}."""
    seen = {}
    for r in runs.values():
        if r.meta["lr"] is None or len(r.steps) == 0:
            continue
        for tag, sel, key in (("a", sel_a, key_a), ("b", sel_b, key_b)):
            if sel(r):
                seen.setdefault(key(r), {"a": set(), "b": set()})[tag].add(hparam(r))
    return {x: d["a"] & d["b"] for x, d in seen.items()}
