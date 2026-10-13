#!/usr/bin/env python3
"""ICLR Figure 5b: the critical batch size grows over training.

For each target accuracy tau in [0.30, 0.50] (1% steps), refit S(N) = S_min (1 + N*/N) along each sweep
(prompts at K=16, prompts at K=64, rollouts at B=128) on the per-cell fastest-LR crossings of tau, and plot
N*(tau) against the training tokens needed to reach tau in the small-batch limit, E_min = S_min * N*, times
the tokens per sequence L. Hollow markers: N* beyond the largest batch in that sweep at that tau.

This is the "FIG 4" section of the full pipeline (critical-batch-rlvr/analysis/iclr_figs/make_figs.py)
made standalone; the series selection rules and the fit are identical.

Usage:
  PLOT iclr_figs/make_cbs_vs_tokens.py --traj csv/traj_all.csv --meta csv/all_runs.jsonl --out paper
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbs_data import load_runs, best_per_cell, fit_mccandlish

# same palette as the other ICLR figures; large type because the panel is placed at half the text width
BLUE, ORANGE, TEAL, INK = "#2B6CB0", "#E07A2F", "#1B9E77", "#222222"
BIG = dict(legend=19, annot=17)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 22, "axes.labelsize": 25, "axes.titlesize": 25, "legend.fontsize": BIG["legend"],
    "xtick.labelsize": 22, "ytick.labelsize": 22,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42,
})

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tau_max", type=float, default=0.50, help="largest target accuracy refit (the main figures' target)")
ap.add_argument("--sustained", type=int, default=1, help="consecutive readings >= tau required for a crossing")
ap.add_argument("--tokens_per_seq", type=float, default=4096.0,
                help="tokens per sequence for the token axis (default: the 1024+3072 context budget); 0 = measured mean length")
ap.add_argument("--stem", default="fig4_cbs_vs_tokens")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
B0, K0 = 128, 16

# ------------------------------------------------------------------ cells (selection rules shared with make_figs.py)
runs = load_runs(args.traj, args.meta)
DENSE_B = {r.meta["B"] for r in runs.values() if r.meta["kind"] == "std" and r.meta["K"] == 64 and "dense" in r.key}
SER = {
    "prompts":   dict(sel=lambda r: r.meta["kind"] == "std" and r.meta["K"] == K0,
                      seqs=lambda c: c[2] * K0, color=BLUE, marker="o", label=f"prompts varied ({K0} rollouts / prompt)"),
    "prompts64": dict(sel=lambda r: (r.meta["kind"] == "std" and r.meta["K"] == 64 and r.meta["upd"]
                                     and not (r.meta["B"] in DENSE_B and "dense" not in r.key)),
                      seqs=lambda c: c[2] * 64, color=TEAL, marker="v", label="prompts varied (64 rollouts / prompt)"),
    "rollouts":  dict(sel=lambda r: r.meta["kind"] == "std" and r.meta["B"] == B0 and (r.meta["upd"] or (r.meta["K"] or 0) <= 16),
                      seqs=lambda c: c[1] * B0, color=ORANGE, marker="s", label=f"rollouts varied ({B0} prompts)"),
}

# tokens per sequence: the fixed context budget, or the mean (prompt + response) length of the best runs at tau_max
if args.tokens_per_seq > 0:
    L_TOK = args.tokens_per_seq
else:
    Ls = []
    for s in SER.values():
        for c, d in best_per_cell(runs, tau=args.tau_max, select=s["sel"], sustained=args.sustained).items():
            if d["S"] is not None and d["best"]:
                L = d["best"]["run"].tokens_per_seq_until(d["S"])
                if np.isfinite(L): Ls.append(L)
    L_TOK = float(np.mean(Ls))
print(f"tokens per sequence: {L_TOK:.0f}")

def fmt_tokens(n):
    if n >= 1e9: return f"{n/1e9:.1f}B"
    if n >= 1e6: return f"{n/1e6:.0f}M" if n >= 1e7 else f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.0f}k"
    return f"{n:.0f}"
def fmt_int(n):
    return f"{int(n):,}" if n < 1e4 else (f"{n/1e3:.0f}k" if n < 1e6 else f"{n/1e6:.0f}M")

# ------------------------------------------------------------------ N*(tau) per sweep
TAUS = np.round(np.arange(0.30, args.tau_max + 0.005, 0.01), 3)
def cbs_vs_tau(sel, seqs_of, min_cells=4):
    out = []
    for tau in TAUS:
        cells = best_per_cell(runs, tau=tau, select=sel, sustained=args.sustained)
        pts = [(seqs_of(c), d["S"]) for c, d in cells.items() if d["S"] is not None]
        if len(pts) < min_cells:
            continue
        x = np.array([p[0] for p in pts], float); S = np.array([p[1] for p in pts], float)
        Smin, Nstar = fit_mccandlish(x, S)
        out.append(dict(tau=tau, Nstar=Nstar, Smin=Smin, Emin=Smin * Nstar, n=len(pts), maxN=float(x.max())))
    return out

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(8.8, 6.2), layout="constrained")
CURVES = {}
for name, s in SER.items():
    cur = cbs_vs_tau(s["sel"], s["seqs"])
    CURVES[name] = cur
    if not cur: continue
    tk = np.array([c["Emin"] for c in cur]) * L_TOK; ns = np.array([c["Nstar"] for c in cur])
    ax.plot(tk, ns, "-", color=s["color"], lw=3.4, alpha=0.5, zorder=2)
    inr = np.array([c["Nstar"] <= c["maxN"] for c in cur])
    ax.plot(tk[inr], ns[inr], s["marker"], color=s["color"], ms=10.5, mec="white", mew=0.9, zorder=3, label=s["label"])
    ax.plot(tk[~inr], ns[~inr], s["marker"], mfc="white", mec=s["color"], mew=2.0, ms=10.5, ls="", zorder=3)
    # K=16 labels go below its curve: with the K=64 point at B=4096 the two series run close together near 1B tokens
    off = (0, -23) if name == "prompts" else (0, 11)
    for c, xx, yy in zip(cur, tk, ns):
        if abs(c["tau"] * 100 - round(c["tau"] * 100)) < 1e-6 and int(round(c["tau"] * 100)) % 5 == 0:
            ax.annotate(f"{c['tau']:.0%}", (xx, yy), xytext=off, textcoords="offset points", ha="center",
                        fontsize=BIG["annot"], color=s["color"])
ax.set_xscale("log"); ax.set_yscale("log", base=2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_int(v)))
xt = [v for v in (1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9) if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]
ax.xaxis.set_major_locator(FixedLocator(xt)); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tokens(v)))
if any((~np.array([c["Nstar"] <= c["maxN"] for c in cur])).any() for cur in CURVES.values() if cur):
    ax.plot([], [], "o", mfc="white", mec=INK, mew=2.0, ms=10.5, ls="", label=r"hollow: $N^*$ beyond largest batch")
ax.set_xlabel("training tokens to reach the target")     # small-batch limit, S_min * N* * L; said in the caption
ax.set_ylabel("critical batch size $N^*$")
ax.set_ylim(ax.get_ylim()[0] / 4.5, ax.get_ylim()[1])
ax.legend(loc="lower right", fontsize=BIG["legend"], handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.out, f"{args.stem}.{ext}"))
print("wrote", args.stem)

with open(os.path.join(args.out, "cbs_vs_tokens_table.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series", "tau", "Nstar_seqs", "Smin", "Emin_seqs", "tokens", "n_cells"])
    for name, cur in CURVES.items():
        for c in cur:
            w.writerow([name, c["tau"], f"{c['Nstar']:.1f}", f"{c['Smin']:.1f}", f"{c['Emin']:.0f}", f"{c['Emin']*L_TOK:.3e}", c["n"]])
for name, cur in CURVES.items():
    if cur:
        print(f"{name:10s} N* at {cur[0]['tau']:.0%}: {cur[0]['Nstar']:8,.0f}   at {cur[-1]['tau']:.0%}: {cur[-1]['Nstar']:8,.0f}"
              f"   ({cur[-1]['Nstar']/cur[0]['Nstar']:.1f}x over {cur[-1]['Emin']/cur[0]['Emin']:.0f}x the tokens)")
