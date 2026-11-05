"""Accuracy against prompts trained on, from target crossings (the alternative to plot_fixed_budget_acc.py).

Reads the per-cell and per-target tables of plot_nstar_vs_target.py (run it first, e.g. with --min-target 0.35 --sweeps n16
--fixed-min-bsz 128), so every number here is the same crossing that enters the N*(target) fits:
  P(B, tau)  = B x S(B, tau)            prompts the fastest LR at batch size B needs to first reach tau (K = 16 sweep)
  P_min(tau) = S_min(tau) N*(tau) / 16  the small-batch limit of the per-target fit S = S_min (1 + N*/N)

Figure 1 (<stem>.png): accuracy vs prompts, one line per batch size (colour bar), the small-batch frontier P_min in black.
  Each line is an envelope over LRs (the fastest LR per target), not one run's learning curve.
Figure 2 (<stem>_overhead.png): data overhead P / P_min against N / N* (N = 16 B), one point per (B, tau), coloured by tau, with the
  McCandlish prediction 1 + N/N* and the 0.2 N* mark (20% more data), so the drop-off is read directly against N*.
Cells whose LR bracket is not closed (status != complete) are hollow; first-interval crossings are left out.

Figure 1 draws only the batch sizes in --bs (default every other power of 4, plus 8192). Targets a batch size first reached at its
first step (crossing floored at 1 step, prompts = B) would stack vertically at x = B; they are replaced by one point at
(B, the step-1 reading of the best post-fix LR at that B, from --curves), where that batch size's line starts (unmarked). The frontier is
drawn last, on top.

Usage: python plot_acc_vs_prompts.py csv/nstar_vs_target_to60 png/acc_vs_prompts.png --max-target 0.55 [--bs 4,16,64,256,1024,4096,16384]
(csv/nstar_vs_target_from35 stops at 50%; the to60 tables reach 60%)
Writes the PNGs, paper/<stem>.pdf, paper/<stem>_overhead.pdf and csv/<stem>_table.csv.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE = "#1f77b4"
TICK, LABEL, LEGEND = 28, 31, 34
K = 16

stem_in, png_out = sys.argv[1], sys.argv[2]
BS = [int(b) for b in (sys.argv[sys.argv.index("--bs") + 1] if "--bs" in sys.argv else "4,16,64,256,1024,4096,8192,16384").split(",")]
CURVES = sys.argv[sys.argv.index("--curves") + 1] if "--curves" in sys.argv else "csv/val_curves_n16.json"
import json
runs = [r for r in json.load(open(CURVES)) if r.get("kl") is not None and abs(r["kl"] - 1e-3) < 1e-12 and r.get("n", K) == K
        and (r.get("fixed") or r["bsz"] < 128)]       # the admission rule of the N* fits (post-fix only at B >= 128)


def step1(B):
    """best step-1 reading over the admitted runs at batch size B (None if no run evaluated step 1)"""
    v = [val for r in runs if r["bsz"] == B for st, val in zip(r["steps"], r["vals"]) if st == 1]
    return max(v) if v else None
T_MAX = float(sys.argv[sys.argv.index("--max-target") + 1]) if "--max-target" in sys.argv else None   # e.g. 0.55 with the to60 tables
cols = pd.read_csv(stem_in + "_columns.csv")
fits = pd.read_csv(stem_in + "_table.csv").dropna(subset=["nstar_complete"])
if T_MAX is not None:
    cols, fits = cols[cols["target"] <= T_MAX + 1e-9], fits[fits["target"] <= T_MAX + 1e-9]
cols = cols[~cols["first_interval"].astype(bool)].copy()
cols["prompts"] = cols["x"] * cols["steps"]
fits["p_min"] = fits["smin_complete"] * fits["nstar_complete"] / K
d = cols.merge(fits[["target", "nstar_complete", "p_min"]], on="target")
d["ratio"] = d["prompts"] / d["p_min"]; d["n_over"] = d["N"] / d["nstar_complete"]
stem = os.path.basename(png_out).replace(".png", "")
d.to_csv(os.path.join(os.path.dirname(os.path.abspath(stem_in)), stem + "_table.csv"), index=False)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)


def new_ax():
    fig, ax = plt.subplots(figsize=(0.53 * 26.0, 8.6), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    return fig, ax


fk = lambda v: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"
pct = FuncFormatter(lambda v, _: f"{v:.0%}")

# ------------------------------------------------------------------ figure 1: accuracy vs prompts, one line per batch size
fig, ax = new_ax()
dsel = d[d["x"].isin(BS)]
d1 = dsel[dsel["steps"] > 1.0 + 1e-6]                            # crossings after the first step
first = {B: step1(B) for B in sorted(dsel.loc[dsel["steps"] <= 1.0 + 1e-6, "x"].unique())}   # batch sizes with first-step crossings
first = {B: v for B, v in first.items() if v is not None}
Bs = sorted(set(d1["x"]) | set(first))
cols_b = plt.get_cmap("plasma")(np.linspace(0.82, 0.0, len(Bs)))   # one discrete colour per batch size, light at small B (around the frontier), dark at large B; no pale yellow end
for B, c in zip(Bs, cols_b):
    g = d1[d1["x"] == B].sort_values("target")
    px, py = list(g["prompts"]), list(g["target"])
    if B in first:      # the line starts at the first-step reading
        px, py = [float(B)] + px, [first[B]] + py
    ax.plot(px, py, lw=3.0, color=c, alpha=0.7, zorder=3)
    op = g[g["status"] != "complete"]
    ax.scatter(op["prompts"], op["target"], s=6 ** 2 * 1.5, facecolors=SURFACE, edgecolors=c, linewidths=1.8, zorder=4)
f = fits.sort_values("target")
ax.plot(f["p_min"], f["target"], ls="-", lw=5.0, color=INK, zorder=6, solid_capstyle="round")   # on top of the curves
ax.set_xscale("log")
xt = [m * 10 ** e for e in range(2, 8) for m in (1, 2, 5)]
x0, x1 = min(d1["prompts"].min(), min(first, default=np.inf)) / 1.3, d1["prompts"].max() * 1.3
ax.set_xlim(min(x0, f["p_min"].min() / 1.3), x1)
ax.xaxis.set_major_locator(FixedLocator([v for v in xt if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]])); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fk(v)))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05)); ax.yaxis.set_major_formatter(pct)   # whole 5% steps (a 2.5% step would round to 43%, 48%)
ax.set_xlabel("prompts trained on", color=INK, fontsize=LABEL); ax.set_ylabel("accuracy reached", color=INK, fontsize=LABEL)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ListedColormap(cols_b), norm=BoundaryNorm(np.arange(len(Bs) + 1), len(Bs))), ax=ax, pad=0.02, aspect=22)
cb.set_ticks(np.arange(len(Bs)) + 0.5); cb.set_ticklabels([f"{b // 1024}k" if b >= 1024 else str(b) for b in Bs])
cb.ax.minorticks_off(); cb.ax.tick_params(labelsize=TICK, colors=INK, length=0); cb.outline.set_visible(False)
cb.set_label(f"prompts per step B  (K = {K})", color=INK, fontsize=LABEL)
ax.legend(handles=[Line2D([], [], ls="-", lw=5.0, color=INK, label="fitted small-batch limit")],
          loc="lower right", frameon=False, fontsize=TICK - 2, labelcolor=INK, handlelength=2.0, borderaxespad=0.2)
fig.savefig(png_out, dpi=170, facecolor=SURFACE); fig.savefig(os.path.join(pdf_dir, stem + ".pdf"), facecolor=SURFACE)
print("wrote", png_out)

# ------------------------------------------------------------------ figure 2: data overhead vs N / N*
fig, ax = new_ax()
taus = sorted(d["target"].unique())
tmap = LinearSegmentedColormap.from_list("tau", ["#9ecae1", BLUE, "#08306b"]); tnorm = Normalize(min(taus), max(taus))
ok, op = d[d["status"] == "complete"], d[d["status"] != "complete"]
ax.scatter(ok["n_over"], ok["ratio"], s=7 ** 2 * 1.5, c=tmap(tnorm(ok["target"])), edgecolors=SURFACE, linewidths=0.6, zorder=4)
ax.scatter(op["n_over"], op["ratio"], s=7 ** 2 * 1.5, facecolors=SURFACE, edgecolors=tmap(tnorm(op["target"])), linewidths=1.6, zorder=4)
xs = np.geomspace(d["n_over"].min() / 1.3, d["n_over"].max() * 1.3, 200)
ax.plot(xs, 1 + xs, ls="-", lw=4.8, color=INK, zorder=5)
ax.axvline(0.2, ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("batch relative to the critical batch,  $N / N^*$", color=INK, fontsize=LABEL)
ax.set_ylabel("prompts needed / small-batch limit", color=INK, fontsize=LABEL)
sm = plt.cm.ScalarMappable(cmap=tmap, norm=tnorm)
cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=22)
cb.set_ticks([t for t in taus if abs(t * 100 - round(t * 100)) < 1e-6 and int(round(t * 100)) % 5 == 0]); cb.ax.yaxis.set_major_formatter(pct)
cb.ax.minorticks_off(); cb.ax.tick_params(labelsize=TICK, colors=INK, length=0); cb.outline.set_visible(False)
cb.set_label("target accuracy", color=INK, fontsize=LABEL)
ax.legend(handles=[Line2D([], [], ls="-", lw=4.8, color=INK, label="$1 + N/N^*$"),
                   Line2D([], [], ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8, label="$N = 0.2\\,N^*$")],
          loc="upper left", frameon=False, fontsize=TICK - 2, labelcolor=INK, handlelength=2.0, borderaxespad=0.2)
out2 = png_out.replace(".png", "_overhead.png")
fig.savefig(out2, dpi=170, facecolor=SURFACE); fig.savefig(os.path.join(pdf_dir, stem + "_overhead.pdf"), facecolor=SURFACE)
print("wrote", out2)
r = np.log(d["ratio"]) - np.log(1 + d["n_over"])
print(f"{len(d)} cells; log-ratio residual vs 1 + N/N*: median {np.median(r):+.3f}, rms {np.sqrt((r ** 2).mean()):.3f}")
