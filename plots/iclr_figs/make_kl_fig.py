#!/usr/bin/env python3
"""Effect of the KL coefficient at K=16, two panels in the style of plots/scripts/plot_kl_compare_clean.py:
  (a) steps to target vs batch size B, the fastest LR per KL, with each KL's Figure 1 fit S = S_min (1 + B*/B) (read from the
      Figure 1 fit table, not refitted) and its fitted critical batch B* as a dashed vertical;
  (b) learning rate vs B: the fastest LR per KL (markers, dashed) and the highest "stable" LR (solid), the band between them shaded;
      LRs above the ceiling that did not qualify are drawn as faint hollow dots.
'stable' is set by --stable: reach (default) = the run crossed tau; hold = and its last 10% of readings still average >= tau;
peak = the original rule from cbs-for-rl (last 10% of readings >= 60% of the run's peak).
Started from Rachit's make_kl_fig.py (cbs-for-rl@onpol-theory-reeval-jun10); his panel (c), the matched-pair trajectories, was dropped.
Usage: python make_kl_fig.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out [--tau 0.5] [--stable reach]
       [--fit-table csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv] [--lr-max 1.2e-4]
Writes fig_kl_stability.{png,pdf} (both panels; the PDF also to plots/paper/) and fig_kl_stability_{a,b}.{png,pdf}; prints the ceiling per B and KL."""
import argparse, os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Paper scale (shared style with the iclr_figs figures and paperstyle.py): this script lays figures out on a 26-inch "line width"
# with 28-34 pt type; drawing it on a 19-inch canvas instead (x 19/26) prints the type at ~8-10 pt and lines at ~1.4 pt when the
# figure is scaled to the 5.5-inch text width, the same as the other paper figures.
PAPER_SCALE = 19.0 / 26.0
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbs_data import load_runs, best_per_cell
ap = argparse.ArgumentParser(); ap.add_argument("--traj", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--stable", choices=["hold", "peak", "reach"], default="reach",
    help="what a 'stable' run must do beyond crossing tau: hold = its last 10%% of readings still average >= tau; "
         "peak = the original rule, last 10%% >= 60%% of its peak; reach = nothing more, crossing tau is enough (default)")
ap.add_argument("--fit-table", default="csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv",
    help="Figure 1 fit table; panel (a) draws each KL's fitted S = S_min (1 + B*/B) and marks B*. '' = points only")
ap.add_argument("--lin", type=float, default=None, help="mark where linear scaling ends, N = LIN N* (the fit is (1 + LIN) x perfect scaling; 0.2 = 20%% above, as in Figures 1-2) instead of B*")
ap.add_argument("--lr-max", type=float, default=1.2e-4, help="LRs above this are left off panel (b); nothing above 1.2e-4 ever reached the target")
a = ap.parse_args()

# ---- style, as in plot_kl_compare_clean.py
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
KLC = {0.001: "#1f77b4", 0.01: "#c2185b"}   # KL 1e-3 blue (Figure 1 K = 16 blue), KL 1e-2 pink
KLN = {0.001: "KL 1e-3", 0.01: "KL 1e-2"}; MK = {0.001: "o", 0.01: "s"}; MS = {0.001: 15.5, 0.01: 13.8}   # paper marker size shared with the LR split and upsampling figures (s = 360 for circles)
TICK, LABEL, TITLE, LEGEND = 28, 31, 32, 27
K0 = 16; KLS = (0.001, 0.01)
runs = load_runs(a.traj, a.meta)
fmt = lambda x: f"{int(x) // 1024}k" if x >= 1024 else str(int(x))


def outcome(r):
    """never / reached_collapsed / reached_short / reached_stable (see --stable)"""
    v = np.asarray(r.vals, float); peak = v.max(); tail = v[-max(3, len(v) // 10):].mean(); S = r.crossing(a.tau)
    if a.stable == "peak":
        collapsed = tail < 0.6 * peak
        if S is None: return "never", S
        if collapsed: return "reached_collapsed", S
        if r.last_step < 1.5 * S: return "reached_short", S
        return "reached_stable", S
    collapsed = (a.stable == "hold") and tail < a.tau
    if S is None: return "never", S
    if r.last_step < 1.5 * S: return "reached_short", S
    if collapsed: return "reached_collapsed", S
    return "reached_stable", S


rows = []
for r in runs.values():
    m = r.meta
    if m["kind"] != "std" or m["K"] != K0 or m["lr"] is None or len(r.steps) < 3 or r.kl is None or m["B"] < 8 or m["lr"] > a.lr_max * 1.001: continue
    o, S = outcome(r); rows.append(dict(B=m["B"], lr=m["lr"], kl=float(r.kl), out=o, S=S))
sel = lambda kl: (lambda r: r.meta["kind"] == "std" and r.meta["K"] == K0 and r.kl is not None and abs(r.kl - kl) < 1e-9)
best, bestS, ceil = {}, {}, {}
for kl in KLS:
    cells = best_per_cell(runs, tau=a.tau, select=sel(kl))
    best[kl] = {c[2]: d["best"]["lr"] for c, d in cells.items() if d["best"] and c[2] >= 8}
    bestS[kl] = {c[2]: d["S"] for c, d in cells.items() if d["S"] is not None and c[2] >= 8}
    ceil[kl] = {}
    for x in rows:
        if x["kl"] == kl and x["out"] in ("reached_stable", "reached_short"): ceil[kl][x["B"]] = max(ceil[kl].get(x["B"], 0), x["lr"])
Bs = sorted({x["B"] for x in rows})


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    major = [x for x in Bs if int(round(np.log2(x))) % 2 == 0]
    ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major])
    ax.set_xticks([x for x in Bs if x not in major], minor=True); ax.set_xticklabels([""] * len([x for x in Bs if x not in major]), minor=True)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel(f"prompts B  (K = {K0})", color=INK, fontsize=LABEL)


# ---------------- (a) steps to target, fastest LR per KL, with the Figure 1 fits
def panel_a(ax):
    fits = pd.read_csv(a.fit_table).set_index("series") if a.fit_table and os.path.exists(a.fit_table) else None
    FIT_KEY = {0.001: "more prompts, K = 16", 0.01: "more prompts, K = 16, KL 1e-2"}
    for row, kl in enumerate(KLS):
        p = sorted(bestS[kl].items()); x = np.array([b for b, _ in p], float); y = np.array([s for _, s in p])
        if fits is not None and FIT_KEY[kl] in fits.index:
            f = fits.loc[FIT_KEY[kl]]; smin, nstar = float(f["s_min"]), float(f["n_star_seqs"]); bstar = nstar / K0
            xs = np.geomspace(x.min() / 1.4, x.max() * 1.4, 200)
            ax.plot(xs, smin * (1 + nstar / (K0 * xs)), ls="-", lw=4.8, color=KLC[kl], zorder=3)
            if a.lin:      # where linear scaling ends (open circle on the fit), labelled B ~ ... at the bottom left, stacked, clear of the curves
                bm = a.lin * bstar
                ax.axvline(bm, ls=(0, (3, 3)), lw=2.6, color=KLC[kl], alpha=0.55, zorder=2.3)
                ax.plot([bm], [smin * (1 + nstar / (K0 * bm))], marker="o", ms=15, mfc=SURFACE, mec=KLC[kl], mew=3.0, ls="", zorder=8)
                bx = a.lin * float(fits.loc[FIT_KEY[KLS[0]], "n_star_seqs"]) / K0      # both labels end left of the leftmost (KL 1e-3) line
                ax.annotate(f"$B \\approx$ {bm:.0f}", (bx, 0.17 - 0.11 * row), xycoords=("data", "axes fraction"), xytext=(-10, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=LEGEND + 2, color=KLC[kl], zorder=10,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
            else:
                ax.axvline(bstar, ls=(0, (3, 3)), lw=2.6, color=KLC[kl], alpha=0.55, zorder=2.3)
                ax.annotate(f"$B^* \\approx$ {bstar / 1000:.1f}k", (bstar, 0.60 - 0.13 * row), xycoords=("data", "axes fraction"), xytext=(-10, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=LEGEND + 2, color=KLC[kl], zorder=10,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        else:
            ax.plot(x, y, lw=4.8, color=KLC[kl], zorder=3)
        ax.scatter(x, y, s=MS[kl] ** 2 * 1.5, marker=MK[kl], color=KLC[kl], edgecolors=SURFACE, linewidths=1.5, zorder=6)
    style(ax); ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)
    hnd = []      # the KL colours are in the shared strip under both panels (fig_kl_stability_legend)
    if fits is not None: hnd.append(Line2D([], [], ls="-", lw=4.8, color=INK2, label="fit  $S = S_\\min\\,(1 + N^*/N)$"))
    if fits is not None and a.lin:
        hnd.append(Line2D([], [], ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8, marker="o", ms=12, mfc=SURFACE, mec=INK2, mew=2.4,
                          label=f"linear scaling ends\n(${a.lin:g} \\times N^*$)"))
    if a.lin:      # B labels sit at the bottom left of their lines, so the legend takes the empty upper right, opaque over the dashed lines
        ax.legend(handles=hnd, loc="upper right", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1.0, fontsize=LEGEND - 2,
                  labelcolor=INK, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35, borderpad=0.3)
    else:
        ax.legend(handles=hnd, loc="lower left", frameon=False, fontsize=LEGEND, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3)


# ---------------- (b) fastest LR vs highest stable LR per KL
STABLE_LABEL = {"reach": "largest LR that reaches target", "hold": "largest LR that holds target", "peak": "highest stable LR"}
FAIL_LABEL = {"reach": "did not reach target", "hold": "did not reach / hold target", "peak": "collapsed / did not reach"}
def panel_b(ax):
    for kl, nudge in zip(KLS, (-0.07, +0.07)):
        bs = np.array(sorted(set(best[kl]) & set(ceil[kl])), float) * 2.0 ** nudge
        yb = np.array([best[kl][int(round(v))] for v in bs / 2.0 ** nudge]); yc = np.array([ceil[kl][int(round(v))] for v in bs / 2.0 ** nudge])
        ax.fill_between(bs, yb, yc, color=KLC[kl], alpha=0.12, lw=0)
        ax.plot(bs, yc, ls="-", lw=4.8, color=KLC[kl], zorder=5)
        ax.plot(bs, yb, ls=(0, (5, 2.5)), lw=4.0, color=KLC[kl], alpha=0.8, zorder=4)
        ax.scatter(bs, yb, s=MS[kl] ** 2 * 1.5, marker=MK[kl], color=KLC[kl], edgecolors=SURFACE, linewidths=1.5, zorder=6)
        for x in rows:   # runs above the ceiling that did not qualify
            if x["kl"] == kl and x["out"] in ("reached_collapsed", "never") and x["lr"] > ceil[kl].get(x["B"], 0):
                ax.scatter([x["B"] * 2.0 ** nudge], [x["lr"]], s=110, marker="o", facecolors="none", edgecolors=KLC[kl], alpha=0.35, linewidths=1.8, zorder=3)
    style(ax); ax.set_ylabel("learning rate", color=INK, fontsize=LABEL); ax.set_xlim(min(Bs) / 1.9, max(Bs) * 1.9)
    lo = min(min(best[kl].values()) for kl in KLS); ax.set_ylim(min(lo / 1.2, 9e-7) / 2.6, ax.get_ylim()[1])   # below the 1e-6 tick: an empty band under the data for the legend
    hnd = [Line2D([], [], ls=(0, (5, 2.5)), lw=4.0, marker="o", ms=12, color=INK2, mec=SURFACE, label="fastest LR"),
           Line2D([], [], ls="-", lw=4.8, color=INK2, label=STABLE_LABEL[a.stable]),
           Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK2, alpha=0.6, mew=1.8, label=FAIL_LABEL[a.stable])]
    ax.legend(handles=hnd, loc="lower right", frameon=False, fontsize=LEGEND - 3, labelcolor=INK, handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.25, handlelength=1.6)      # inside panel (b), in its empty lower-right corner: a strip under both panels would read as applying to (a) too


paper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper"); os.makedirs(paper, exist_ok=True)
# sub-figures for a LaTeX subfigure layout (a: 0.53, b: 0.46 of the line width): not cropped, same canvas height and top margin, so the
# two top-align when LaTeX sets them side by side (as for the LR split figure)
for tag, fn, wfrac in (("a", panel_a, 0.53), ("b", panel_b, 0.46)):
    fig, ax = plt.subplots(figsize=(PAPER_SCALE * (wfrac * 26.0), PAPER_SCALE * (8.6)), facecolor=SURFACE, layout="constrained"); fn(ax)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_kl_stability_{tag}.{ext}"), dpi=170, facecolor=SURFACE)
    fig.savefig(os.path.join(paper, f"fig_kl_stability_{tag}.pdf"), facecolor=SURFACE)
    plt.close(fig)
fig, (axl, axr) = plt.subplots(1, 2, figsize=(PAPER_SCALE * (26.0), PAPER_SCALE * (8.6)), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.22, wspace=0.22)
panel_a(axl); panel_b(axr)
for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_kl_stability.{ext}"), dpi=170, facecolor=SURFACE, bbox_inches="tight")
# shared legend strip under both sub-figures (the KL colours apply to (a) and (b)); fixed canvas at the full line width, checked to fit
KL_HND = [Line2D([], [], marker=MK[kl], ls="-", lw=4.8, ms=MS[kl], color=KLC[kl], mec=SURFACE, label=KLN[kl]) for kl in KLS]
fl = plt.figure(figsize=(PAPER_SCALE * 26.0, PAPER_SCALE * 1.5), facecolor=SURFACE)
leg = fl.legend(handles=KL_HND, loc="center", ncol=len(KLS), frameon=False, fontsize=LEGEND, labelcolor=INK, handlelength=2.2,
                handletextpad=0.6, columnspacing=3.0)
fl.canvas.draw(); bb = leg.get_window_extent()
if bb.x0 < -1 or bb.x1 > fl.bbox.x1 + 1 or bb.y0 < -1 or bb.y1 > fl.bbox.y1 + 1:
    raise SystemExit(f"KL legend strip {bb.bounds} does not fit in {fl.bbox.bounds}")
for ext in ("pdf", "png"): fl.savefig(os.path.join(a.out, f"fig_kl_stability_legend.{ext}"), dpi=170, facecolor=SURFACE)
fl.savefig(os.path.join(paper, "fig_kl_stability_legend.pdf"), facecolor=SURFACE)
fig.savefig(os.path.join(paper, "fig_kl_stability.pdf"), facecolor=SURFACE, bbox_inches="tight")   # paper copy, next to the other paper-ready PDFs
print("wrote", os.path.join(a.out, "fig_kl_stability.png"), "and paper/fig_kl_stability.pdf")
for kl in KLS:
    print(KLN[kl], f"highest stable LR (rule: {a.stable}):", {b: f"{ceil[kl][b]:g}" for b in sorted(ceil[kl])})
