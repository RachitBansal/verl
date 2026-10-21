#!/usr/bin/env python3
"""Panel (b) of the KL-stability figure under a common horizon.

make_kl_fig.py calls a run "stable" if it reached the target and (--stable hold) its last 10% of readings still average at least
the target; by default (--stable reach) reaching the target is enough; --stable peak is the original rule (last 10% >= 60% of the run's peak). That
verdict depends on how long the run was allowed to continue past its crossing, which differs by batch size, code version and
KL (KL 1e-3 runs were watched ~3.4x their crossing on average, KL 1e-2 runs ~2.5x). Here every run is truncated at H x its own
crossing step before the verdict is taken (runs that never crossed are truncated at H x the cell's fastest crossing), so both
KLs get the same opportunity to collapse. Left: the original panel (b); right: the same with the horizon H.

Usage: python make_kl_fig_horizon.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out [--tau 0.5] [--horizon 2.5]
Writes fig_kl_stability_b_horizon{H}.{pdf,png} and prints the per-B ceilings under each rule.
"""
import argparse, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbs_data import load_runs, best_per_cell
ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--horizon", type=float, default=2.5)
ap.add_argument("--stable", choices=["hold", "peak", "reach"], default="reach", help="stability rule, as in make_kl_fig.py (hold = tail still >= tau; peak = tail >= 60%% of peak; reach)")
a = ap.parse_args()
KLC = {0.001: "#1f77b4", 0.01: "#c2185b"}; KLN = {0.001: "KL 1e-3", 0.01: "KL 1e-2"}; INK = "#222222"   # KL 1e-3 blue (Figure 1 K = 16 blue), KL 1e-2 pink
plt.rcParams.update({"font.family": "sans-serif", "font.size": 22, "axes.labelsize": 25, "legend.fontsize": 19, "axes.titlesize": 24,
    "xtick.labelsize": 22, "ytick.labelsize": 22, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42})
runs = load_runs(a.traj, a.meta); K0 = 16
fmt = lambda v: f"{int(v):,}" if v < 1000 else f"{round(v/1e3)}k"


def outcome(r, cap=None):
    """make_kl_fig's rule, applied to the run truncated at step `cap` (None = the whole run)."""
    st, v = np.asarray(r.steps, float), np.asarray(r.vals, float)
    if cap is not None:
        m = st <= cap
        if m.sum() < 3: return "short", None
        st, v = st[m], v[m]
    S = r.crossing(a.tau)
    if S is not None and cap is not None and S > cap: S = None
    peak = v.max(); tail = v[-max(3, len(v) // 10):].mean()
    if a.stable == "peak":
        collapsed = tail < 0.6 * peak
        if S is None: return "never", S
        if collapsed: return "reached_collapsed", S
        if st[-1] < 1.5 * S: return "reached_short", S
        return "reached_stable", S
    collapsed = (a.stable == "hold") and tail < a.tau
    if S is None: return "never", S
    if st[-1] < 1.5 * S: return "reached_short", S
    if collapsed: return "reached_collapsed", S
    return "reached_stable", S


rows = [r for r in runs.values() if r.meta["kind"] == "std" and r.meta["K"] == K0 and r.meta["lr"] is not None and len(r.steps) >= 3 and r.kl is not None]
Bs = sorted({r.meta["B"] for r in rows if r.meta["B"] >= 8})


def ceilings(H):
    """per KL: fastest LR per B and the highest LR whose (truncated) run reached and stayed; plus the collapsed/never points"""
    out = {}
    for kl in (0.001, 0.01):
        cells = best_per_cell(runs, tau=a.tau, select=lambda r, kl=kl: r.meta["kind"] == "std" and r.meta["K"] == K0 and r.kl is not None and abs(r.kl - kl) < 1e-9)
        best = {c[2]: d["best"]["lr"] for c, d in cells.items() if d["best"] and c[2] >= 8}
        bestS = {c[2]: d["S"] for c, d in cells.items() if d["S"] is not None and c[2] >= 8}
        ceil, bad = {}, []
        for r in rows:
            B = r.meta["B"]
            if abs(r.kl - kl) > 1e-9 or B < 8: continue
            cap = None
            if H is not None:
                S_own = r.crossing(a.tau); ref = S_own if S_own is not None else bestS.get(B)
                cap = H * ref if ref else None
            o, S = outcome(r, cap)
            if o in ("reached_stable", "reached_short"): ceil[B] = max(ceil.get(B, 0), r.meta["lr"])
            elif o in ("reached_collapsed", "never"): bad.append((B, r.meta["lr"]))
        out[kl] = dict(best=best, ceil=ceil, bad=[(B, lr) for B, lr in bad if lr > ceil.get(B, 0)])
    return out


def panel_b(ax, res, title):
    for kl in (0.001, 0.01):
        best, ceil, bad = res[kl]["best"], res[kl]["ceil"], res[kl]["bad"]
        bs = np.array(sorted(set(best) & set(ceil)), float); yb = np.array([best[int(v)] for v in bs]); yc = np.array([ceil[int(v)] for v in bs])
        ax.fill_between(bs, yb, yc, color=KLC[kl], alpha=0.12, lw=0)
        ax.plot(bs, yc, "-", color=KLC[kl], lw=3.6, zorder=3); ax.plot(bs, yb, "o" if kl == 0.001 else "D", color=KLC[kl], ms=11, mec="white", mew=1.2, zorder=4)
        ax.plot(bs, yb, "--", color=KLC[kl], lw=2.0, alpha=0.7, zorder=2)
        for B, lr in bad: ax.plot([B], [lr], "x", color=KLC[kl], ms=10, mew=2.0, alpha=0.6, zorder=2)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.xaxis.set_major_locator(FixedLocator(Bs[::2])); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticklabels([fmt(b) for b in Bs[::2]])
    ax.set_xlabel("prompts per step $B$   ($K=16$)"); ax.set_ylabel(r"learning rate $\eta$"); ax.set_title(title, loc="left")


orig, trunc = ceilings(None), ceilings(a.horizon)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharey=True)
panel_b(axes[0], orig, "as run (horizon varies per run)"); panel_b(axes[1], trunc, f"every run cut at {a.horizon:g}× its crossing")
hnd = [Line2D([], [], marker="o", ls="--", color=INK, lw=2.0, ms=10, mec="white", label="fastest LR"),
       Line2D([], [], color=INK, lw=3.6, label="highest stable LR"),
       Line2D([], [], marker="x", ls="", color=INK, ms=10, mew=2.0, alpha=0.6, label="collapsed / did not reach")]
hnd += [Line2D([], [], color=KLC[kl], lw=3.6, label=KLN[kl]) for kl in (0.001, 0.01)]
axes[0].legend(handles=hnd, loc="upper left", handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
fig.tight_layout(w_pad=2.5)
tag = f"{a.horizon:g}".replace(".", "p") + ("" if a.stable == "reach" else f"_{a.stable}")
for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_kl_stability_b_horizon{tag}.{ext}"))
print("wrote", os.path.join(a.out, f"fig_kl_stability_b_horizon{tag}.png"))
for name, res in (("as run", orig), (f"horizon {a.horizon:g}x", trunc)):
    for kl in (0.001, 0.01):
        print(f"{name:14s} {KLN[kl]}: ceiling " + ", ".join(f"{b}:{res[kl]['ceil'][b]:g}" for b in sorted(res[kl]["ceil"])))
