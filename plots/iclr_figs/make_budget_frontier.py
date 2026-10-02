#!/usr/bin/env python3
"""Budget frontier: what each (B, K) cell costs in compute and in data, against wall-clock.

Every cell in the three steps-to-target sweeps (prompts at K=16, prompts at K=64, rollouts at
B=128) is re-plotted as the totals it consumed before first reaching the target:

  (a) total sequences generated  E = S * B * K   (compute)      vs  steps to target S  (wall-clock)
  (b) total prompt draws         P = S * B       (data)         vs  steps to target S

Under the McCandlish fit S = S_min (1 + N*/N) used elsewhere, (a) is the hyperbola
(S/S_min - 1)(E/E_min - 1) = 1 with E_min = S_min N*, drawn dashed; (b) is the same curve
divided by K for a prompt sweep, and the straight line P = 128 S for the rollout sweep.

Usage:
  .venv/bin/python analysis/iclr_figs/make_budget_frontier.py \
      --traj analysis/iclr_figs/data/traj_all.csv --meta analysis/iclr_figs/data/all_runs.jsonl \
      --out ../-ICLR-cbs-for-rl/iclr/figures/cbs_iclr
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbs_data import load_runs, best_per_cell, fit_mccandlish

# same palette / type as make_figs.py so the figure sits next to figs 1-4
BLUE, ORANGE, TEAL, GREY, INK, INK2 = "#2B6CB0", "#E07A2F", "#1B9E77", "#9A9A9A", "#222222", "#55534f"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18, "legend.fontsize": 15,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42,
})

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--n_prompts", type=int, default=17917, help="distinct training prompts (dapo.parquet rows)")
ap.add_argument("--stem", default="fig_budget_frontier")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
TAU, B0, K0 = a.tau, 128, 16

# ------------------------------------------------------------------ cells (selection rules copied from make_figs.py)
runs = load_runs(a.traj, a.meta)
DENSE_B = {r.meta["B"] for r in runs.values() if r.meta["kind"] == "std" and r.meta["K"] == 64 and "dense" in r.key}
SEL = {
    "prompts":   lambda r: r.meta["kind"] == "std" and r.meta["K"] == K0,
    "rollouts":  lambda r: r.meta["kind"] == "std" and r.meta["B"] == B0 and (r.meta["upd"] or (r.meta["K"] or 0) <= 16),
    "prompts64": lambda r: (r.meta["kind"] == "std" and r.meta["K"] == 64 and r.meta["upd"]
                            and not (r.meta["B"] in DENSE_B and "dense" not in r.key)),
}
STYLE = {
    "prompts":   dict(color=BLUE,   marker="o", ms=10, label=f"prompts varied ({K0} rollouts / prompt)"),
    "prompts64": dict(color=TEAL,   marker="v", ms=10, label="prompts varied (64 rollouts / prompt)"),
    "rollouts":  dict(color=ORANGE, marker="s", ms=9,  label=f"rollouts varied ({B0} prompts)"),
}
SER = {}
for name, sel in SEL.items():
    cells = best_per_cell(runs, tau=TAU, select=sel)
    pts = []
    for c, d in cells.items():
        if d["S"] is None:
            continue
        B = c[2] if name != "rollouts" else B0
        K = {"prompts": K0, "prompts64": 64}.get(name, c[1])
        pts.append(dict(B=B, K=K, N=B * K, S=d["S"], E=d["S"] * B * K, P=d["S"] * B, lr=d["best"]["lr"]))
    pts.sort(key=lambda p: p["N"])
    Smin, Nstar = fit_mccandlish([p["N"] for p in pts], [p["S"] for p in pts])
    SER[name] = dict(pts=pts, Smin=Smin, Nstar=Nstar, Emin=Smin * Nstar, **STYLE[name])
    print(f"{name:10s} {len(pts):2d} cells  S_min={Smin:6.2f}  N*={Nstar:8,.0f} seqs  E_min=S_min N*={Smin*Nstar/1e3:6.0f}k seqs"
          f"  | measured min E={min(p['E'] for p in pts)/1e3:5.0f}k  min P={min(p['P'] for p in pts)/1e3:5.1f}k")

with open(os.path.join(a.out, a.stem + "_table.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series", "B", "K", "seqs_per_step", "steps", "total_seqs", "total_prompt_draws", "best_lr"])
    for name, s in SER.items():
        for p in s["pts"]:
            w.writerow([name, p["B"], p["K"], p["N"], f"{p['S']:.2f}", f"{p['E']:.0f}", f"{p['P']:.0f}", p["lr"]])

# ------------------------------------------------------------------ numbers quoted in the text
def at(name, **kw):
    return next(p for p in SER[name]["pts"] if all(p[k] == v for k, v in kw.items()))
p16, p64, ro = SER["prompts"]["pts"], SER["prompts64"]["pts"], SER["rollouts"]["pts"]
E_flat16 = [p for p in p16 if p["E"] <= 1.5 * min(q["E"] for q in p16)]
P_flat64 = [p for p in p64 if p["P"] <= 1.5 * min(q["P"] for q in p64)]
print(f"\nK=16 prompt sweep: E within 1.5x of its minimum for B in {E_flat16[0]['B']}..{E_flat16[-1]['B']}"
      f" (steps {E_flat16[0]['S']:.0f} -> {E_flat16[-1]['S']:.0f})")
print(f"K=64 prompt sweep: P within 1.5x of its minimum for B in {P_flat64[0]['B']}..{P_flat64[-1]['B']}")
sharedB = sorted({p["B"] for p in p16} & {p["B"] for p in p64})
rP = [at("prompts", B=b)["P"] / at("prompts64", B=b)["P"] for b in sharedB]
sharedN = sorted({p["N"] for p in p16} & {p["N"] for p in p64})
rE = [at("prompts64", N=n)["E"] / at("prompts", N=n)["E"] for n in sharedN]
print(f"matched B: K=16 spends {np.median(rP):.2f}x the prompts of K=64 (median; range {min(rP):.2f}-{max(rP):.2f})")
print(f"matched N: K=64 spends {np.median(rE):.2f}x the sequences of K=16 (median; range {min(rE):.2f}-{max(rE):.2f})")
Emin_roll = min(ro, key=lambda p: p["E"]); print(f"rollout sweep: E minimal at K={Emin_roll['K']} ({Emin_roll['E']/1e3:.0f}k seqs)")
r16, r128 = at("rollouts", K=16), at("rollouts", K=128)
print(f"rollout sweep K=16 -> K=128: steps {r16['S']:.0f} -> {r128['S']:.0f}, prompts {r16['P']/1e3:.1f}k -> {r128['P']/1e3:.1f}k,"
      f" sequences {r16['E']/1e3:.0f}k -> {r128['E']/1e3:.0f}k")

# ------------------------------------------------------------------ figure
def fmt_int(v):
    return f"{int(v):,}" if v < 1e4 else (f"{v/1e3:.0f}k" if v < 1e6 else f"{v/1e6:g}M")

fig, (axE, axP) = plt.subplots(1, 2, figsize=(15.4, 6.6), layout="constrained")

def frontier(ax, s, ykey):
    """Dashed McCandlish curve in (S, E) or (S, P) coordinates, then the measured cells."""
    pts = s["pts"]; Ns = np.array([p["N"] for p in pts], float)
    g = np.geomspace(Ns.min() / 1.5, Ns.max() * 1.5, 300)
    S = s["Smin"] * (1 + s["Nstar"] / g); E = s["Smin"] * (g + s["Nstar"])
    if ykey == "E":
        y = E
    elif pts[0]["B"] == pts[-1]["B"]:          # rollout sweep: fixed B, so P = B * S exactly
        y = B0 * S
    else:                                        # prompt sweep: fixed K, so P = E / K
        y = E / pts[0]["K"]
    ax.plot(S, y, "--", color=s["color"], lw=1.6, alpha=0.9, zorder=2)
    xs = [p["S"] for p in pts]; ys = [p[ykey] for p in pts]
    ax.plot(xs, ys, s["marker"], color=s["color"], ms=s["ms"], mec="white", mew=1.2, ls="", zorder=4)

def label(ax, s, p, ykey, text, dx, dy, ha="center", va="center"):
    ax.annotate(text, (p["S"], p[ykey]), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=13, color=s["color"])

for ax, ykey in ((axE, "E"), (axP, "P")):
    for name in ("rollouts", "prompts64", "prompts"):
        frontier(ax, SER[name], ykey)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_int(v)))
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('steps to target  (wall-clock)')
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))

def fmt_B(b):
    return f"{b // 1024}k" if b >= 1024 else str(b)

# (a) compute
axE.set_ylabel('sequences to target  (compute)')
axE.yaxis.set_major_locator(FixedLocator([3e5, 5e5, 1e6, 2e6]))
axE.set_ylim(2.3e5, 2.6e6); axE.set_xlim(4, 9000)
s16, s64, sro = SER["prompts"], SER["prompts64"], SER["rollouts"]
label(axE, s16, p16[0], "E", f"$B={p16[0]['B']}$", 0, 14)
label(axE, s16, p16[-1], "E", f"$B={fmt_B(p16[-1]['B'])}$", 9, -6, ha="left", va="top")
label(axE, s64, p64[0], "E", f"$B={p64[0]['B']}$", 0, 14)
label(axE, s64, p64[-1], "E", f"$B={fmt_B(p64[-1]['B'])}$", 10, 0, ha="left")
label(axE, sro, ro[0], "E", f"$K={ro[0]['K']}$", 0, 14)
label(axE, sro, ro[-1], "E", f"$K={ro[-1]['K']}$", 10, 0, ha="left")
label(axE, sro, Emin_roll, "E", f"$K={Emin_roll['K']}$", 0, -17)
axE.set_title("(a) compute: fewer rollouts per prompt is cheaper", loc="left", color=INK, pad=10)

# (b) data
axP.set_ylabel('prompts to target  (data)')
axP.yaxis.set_major_locator(FixedLocator([1e4, 2e4, 5e4, 1e5, 2e5]))
axP.set_ylim(5.5e3, 3.2e5); axP.set_xlim(4, 9000)
axP.axhline(a.n_prompts, color=INK2, lw=1.4, ls=(0, (5, 4)), zorder=1)
axP.text(8500, a.n_prompts / 1.06, f"one pass over the\ntraining set ({a.n_prompts/1e3:.1f}k)",
         ha="right", va="top", fontsize=13, color=INK2, linespacing=1.1)
label(axP, s16, p16[0], "P", f"$B={p16[0]['B']}$", 0, 14)
label(axP, s16, p16[-1], "P", f"$B={fmt_B(p16[-1]['B'])}$", 10, 2, ha="left")
label(axP, s64, p64[0], "P", f"$B={p64[0]['B']}$", 0, -18)
label(axP, s64, p64[-1], "P", f"$B={fmt_B(p64[-1]['B'])}$", 0, -18)
label(axP, sro, ro[0], "P", f"$K={ro[0]['K']}$", -12, 0, ha="right")
label(axP, sro, ro[-1], "P", f"$K={ro[-1]['K']}$", 0, -18)
axP.set_title("(b) data: more rollouts per prompt spends fewer prompts", loc="left", color=INK, pad=10)

hnd = [Line2D([], [], marker=SER[n]["marker"], color=SER[n]["color"], ms=SER[n]["ms"], lw=0, mec="white", label=SER[n]["label"])
       for n in ("prompts", "prompts64", "rollouts")]
hnd.append(Line2D([], [], ls="--", color=INK, lw=1.6, label=r"McCandlish fit  $S = S_{\min}(1 + N^*/N)$"))
fig.legend(handles=hnd, loc="outside lower center", ncol=2, handletextpad=0.4, columnspacing=2.0, labelspacing=0.35)

fig.get_layout_engine().set(w_pad=0.25, wspace=0.08)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, f"{a.stem}.{ext}"))
print("\nwrote", a.stem)

# ---- single-panel exports for the paper (subfigure captions carry the titles; legend lives in panel a)
plt.close(fig)
for tag, ykey in (("a", "E"), ("b", "P")):
    fg, ax = plt.subplots(figsize=(7.4, 5.6))
    for name in ("rollouts", "prompts64", "prompts"):
        frontier(ax, SER[name], ykey)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_int(v)))
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000])); ax.set_xlim(4, 9000)
    ax.set_xlabel('steps to target  (wall-clock)')
    if ykey == "E":
        ax.set_ylabel('sequences to target  (compute)')
        ax.yaxis.set_major_locator(FixedLocator([3e5, 5e5, 1e6, 2e6])); ax.set_ylim(2.3e5, 4.5e6)
        label(ax, s16, p16[0], "E", f"$B={p16[0]['B']}$", 0, 14)
        label(ax, s16, p16[-1], "E", f"$B={fmt_B(p16[-1]['B'])}$", 9, -6, ha="left", va="top")
        label(ax, s64, p64[0], "E", f"$B={p64[0]['B']}$", 0, 14)
        label(ax, s64, p64[-1], "E", f"$B={fmt_B(p64[-1]['B'])}$", 10, 0, ha="left")
        label(ax, sro, ro[0], "E", f"$K={ro[0]['K']}$", 0, 14)
        label(ax, sro, ro[-1], "E", f"$K={ro[-1]['K']}$", 10, 0, ha="left")
        label(ax, sro, Emin_roll, "E", f"$K={Emin_roll['K']}$", 0, -17)
        ax.legend(handles=hnd, loc="upper right", fontsize=13, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    else:
        ax.set_ylabel('prompts to target  (data)')
        ax.yaxis.set_major_locator(FixedLocator([1e4, 2e4, 5e4, 1e5, 2e5])); ax.set_ylim(5.5e3, 3.2e5)
        ax.axhline(a.n_prompts, color=INK2, lw=1.4, ls=(0, (5, 4)), zorder=1)
        ax.text(8500, a.n_prompts / 1.06, f"one pass over the\ntraining set ({a.n_prompts/1e3:.1f}k)",
                ha="right", va="top", fontsize=13, color=INK2, linespacing=1.1)
        label(ax, s16, p16[0], "P", f"$B={p16[0]['B']}$", 0, 14)
        label(ax, s16, p16[-1], "P", f"$B={fmt_B(p16[-1]['B'])}$", 10, 2, ha="left")
        label(ax, s64, p64[0], "P", f"$B={p64[0]['B']}$", 0, -18)
        label(ax, s64, p64[-1], "P", f"$B={fmt_B(p64[-1]['B'])}$", 0, -18)
        label(ax, sro, ro[0], "P", f"$K={ro[0]['K']}$", -12, 0, ha="right")
        label(ax, sro, ro[-1], "P", f"$K={ro[-1]['K']}$", 0, -18)
    fg.tight_layout()
    for ext in ("pdf", "png"):
        fg.savefig(os.path.join(a.out, f"{a.stem}_{tag}.{ext}"))
    plt.close(fg); print("wrote", f"{a.stem}_{tag}")
