#!/usr/bin/env python3
"""Effect of the KL coefficient at K=16, two panels in the style of plots/scripts/plot_kl_compare_clean.py:
  (a) steps to target vs batch size B, the fastest LR per KL, with each KL's Figure 1 fit S = S_min (1 + B*/B) (read from the
      Figure 1 fit table, not refitted) and its fitted critical batch B* as a dashed vertical;
  (b) the LR grid (as the upsampling figure's right panel): every LR tried at each B for both KLs, filled = reached the target, hollow =
      did not reach it after training at least as long as the fastest LR, black ring = fastest LR. --band restores the earlier panel: the
      fastest LR per KL (markers, dashed) and the highest "stable" LR (solid), the band between them shaded, failing LRs as faint hollow dots.
'stable' is set by --stable: reach (default) = the run crossed tau; hold = and its last 10% of readings still average >= tau;
peak = the original rule from cbs-for-rl (last 10% of readings >= 60% of the run's peak).
Started from Rachit's make_kl_fig.py (cbs-for-rl@onpol-theory-reeval-jun10); his panel (c), the matched-pair trajectories, was dropped.
Usage: python make_kl_fig.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out [--tau 0.5] [--stable reach]
       [--fit-table csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv] [--lr-max 1.2e-4] [--lin 0.2] [--floor] [--panel-b fastest|grid|band] [--height 6.6]
Writes fig_kl_stability.{png,pdf} (both panels and the shared KL legend; the PDF also to plots/paper/), fig_kl_stability_{a,b}.{png,pdf} and
fig_kl_stability_legend.{png,pdf} (the KL legend alone, one row); prints the ceiling per B and KL."""
import argparse, os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
ap.add_argument("--floor", action="store_true", help="panel (b) also draws the smallest LR that reaches the target (dotted) and shades the window from it to the largest")
ap.add_argument("--lin", type=float, default=None, help="mark where linear scaling ends, N = LIN N* (the fit is (1 + LIN) x perfect scaling; 0.2 = 20%% above, as in Figures 1-2) instead of B*")
ap.add_argument("--panel-b", choices=["fastest", "grid", "band"], default="fastest",
    help="fastest (default) = the fastest LR per KL as lines; grid = every LR near the "
         "fastest (see --near), ring = fastest; band = fastest, largest (and with --floor smallest) LR that reaches, band shaded")
ap.add_argument("--near", type=int, default=1, help="LR grid: show only the fastest LR and this many nearest tested LRs on each side (-1 = every LR)")
ap.add_argument("--height", type=float, default=6.6, help="canvas height in inches (the upsampling figure's height; was 8.6)")
ap.add_argument("--lr-max", type=float, default=1.2e-4, help="LRs above this are left off panel (b); nothing above 1.2e-4 ever reached the target")
a = ap.parse_args(); a.near = None if a.near < 0 else a.near

# ---- style, as in plot_kl_compare_clean.py
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
KLC = {0.001: "#1f77b4", 0.01: "#c2185b"}   # KL 1e-3 blue (Figure 1 K = 16 blue), KL 1e-2 pink
KLN = {0.001: "KL 1e-3", 0.01: "KL 1e-2"}; MK = {0.001: "o", 0.01: "s"}; MS = {0.001: 15.5, 0.01: 13.8}   # paper marker size shared with the LR split and upsampling figures (s = 360 for circles)
TICK, LABEL, TITLE, LEGEND = 28, 31, 32, 34
K0 = 16; KLS = (0.001, 0.01)
runs = load_runs(a.traj, a.meta)
fmt = lambda x: f"{int(x) // 1024}k" if x >= 1024 else str(int(x))


KL_HANDLES = lambda: [Line2D([], [], ls="-", lw=4.0, marker=MK[kl], ms=MS[kl], color=KLC[kl], mec=SURFACE, label=KLN[kl]) for kl in KLS]   # the shared KL legend


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
    o, S = outcome(r); rows.append(dict(B=m["B"], lr=m["lr"], kl=float(r.kl), out=o, S=S, last=r.last_step))
sel = lambda kl: (lambda r: r.meta["kind"] == "std" and r.meta["K"] == K0 and r.kl is not None and abs(r.kl - kl) < 1e-9)
best, bestS, ceil, floor = {}, {}, {}, {}
for kl in KLS:
    cells = best_per_cell(runs, tau=a.tau, select=sel(kl))
    best[kl] = {c[2]: d["best"]["lr"] for c, d in cells.items() if d["best"] and c[2] >= 8}
    bestS[kl] = {c[2]: d["S"] for c, d in cells.items() if d["S"] is not None and c[2] >= 8}
    ceil[kl], floor[kl] = {}, {}
    for x in rows:
        if x["kl"] == kl and x["out"] in ("reached_stable", "reached_short"):
            ceil[kl][x["B"]] = max(ceil[kl].get(x["B"], 0), x["lr"]); floor[kl][x["B"]] = min(floor[kl].get(x["B"], np.inf), x["lr"])
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
                ax.annotate(f"$B \\approx$ {bm:.0f}", (bx, 0.21 - 0.145 * row), xycoords=("data", "axes fraction"), xytext=(-10 - 40 * (row == 0), 0),   # the upper (KL 1e-3) label a little further left
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
    hnd = [] if a.panel_b != "band" else [Line2D([], [], marker=MK[kl], ls="", ms=MS[kl], color=KLC[kl], mec=SURFACE, label=KLN[kl]) for kl in KLS]   # the KLs are in the shared legend
    if fits is not None: hnd.append(Line2D([], [], ls="-", lw=4.8, color=INK2, label="fit $S = S_\\min(1 + N^*/N)$"))
    if fits is not None and a.lin:
        hnd.append(Line2D([], [], ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8, marker="o", ms=15, mfc=SURFACE, mec=INK2, mew=3.0,
                          label=f"linear scaling ends (${a.lin:g}\\,N^*$)"))
    if a.lin:      # B labels sit at the bottom left of their lines, so the legend takes the empty upper right, opaque over the dashed lines
        ax.legend(handles=hnd, loc="upper right", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1.0, fontsize=LEGEND - 4,
                  labelcolor=INK, handletextpad=0.6, borderaxespad=0.2, borderpad=0.3)
    else:
        ax.legend(handles=hnd, loc="lower left", frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3)


# ---------------- (b) the LR grid, as the right panel of plot_rollouts_vs_downsample_clean.py: every LR tried at each B for both KLs side
# by side; filled = reached the target, hollow = did not reach it after training at least as long as the fastest LR at that B and KL,
# x = too early to tell (drawn only where the fastest LR is not yet bracketed by definite neighbours); black ring = fastest LR
from matplotlib.legend_handler import HandlerTuple
MISSING = "#9a9893"
def grid_ladder(kl):
    """one row per (B, LR): verdict best / hit / never / inconclusive against the fastest crossing at that B"""
    g = {}
    for x in rows:
        if x["kl"] != kl: continue
        c = g.setdefault((x["B"], x["lr"]), dict(S=None, last=0))
        if x["S"] is not None: c["S"] = x["S"] if c["S"] is None else min(c["S"], x["S"])
        c["last"] = max(c["last"], x["last"])
    out = []
    for B in sorted({b for b, _ in g}):
        if B not in best[kl]: continue      # no crossing at this B
        ref = bestS[kl][B]; col = []
        for (b, lr), c in sorted(g.items()):
            if b != B: continue
            v = ("best" if abs(lr - best[kl][B]) < 1e-12 else "hit") if c["S"] is not None else ("never" if c["last"] >= ref else "inconclusive")
            col.append(dict(B=B, lr=lr, v=v))
        i = next(j for j, r in enumerate(col) if r["v"] == "best")
        definite = lambda j: 0 <= j < len(col) and col[j]["v"] in ("hit", "never")
        if definite(i - 1) and definite(i + 1): col = [r for r in col if r["v"] != "inconclusive"]
        if a.near is not None:      # only the fastest LR and its nearest tested neighbours on each side
            i = next(j for j, r in enumerate(col) if r["v"] == "best"); col = col[max(0, i - a.near):i + a.near + 1]
        out += col
    return out


def panel_b_fastest(ax):
    """the fastest LR per KL vs B, lines with markers (nudged apart so equal LRs stay visible)"""
    style(ax)
    for kl, nudge in zip(KLS, (-0.06, +0.06)):
        bs = sorted(best[kl]); x = np.array(bs, float) * 2.0 ** nudge; y = [best[kl][b] for b in bs]
        ax.plot(x, y, ls="-", lw=4.0, color=KLC[kl], alpha=0.85, zorder=4)
        ax.scatter(x, y, s=MS[kl] ** 2 * 1.5, marker=MK[kl], color=KLC[kl], edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ax.set_ylabel("fastest learning rate", color=INK, fontsize=LABEL); ax.set_xlim(min(Bs) / 1.6, max(Bs) * 1.6)
    lo, hi = ax.get_ylim(); yt = [m * 10.0 ** e for e in (-7, -6, -5, -4) for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]   # the range spans < 2 decades, so label 1-2-5
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(yt)); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 10 ** np.floor(np.log10(v)):.0f}e{int(np.floor(np.log10(v)))}"))



def panel_b(ax):
    if a.panel_b == "band": return panel_b_band(ax)
    if a.panel_b == "fastest": return panel_b_fastest(ax)
    style(ax)
    any_inc = False
    for kl, nudge in zip(KLS, (-0.15, +0.15)):
        lad = grid_ladder(kl); c = KLC[kl]; s = MS[kl] ** 2 * 1.5
        pick = lambda *vs: ([r["B"] * 2.0 ** nudge for r in lad if r["v"] in vs], [r["lr"] for r in lad if r["v"] in vs])
        ax.scatter(*pick("hit", "best"), s=s, marker=MK[kl], color=c, edgecolors=SURFACE, linewidths=1.5, zorder=5)
        ax.scatter(*pick("never"), s=s, marker=MK[kl], facecolors="none", edgecolors=c, linewidths=2.6, zorder=5)
        xi, yi = pick("inconclusive"); any_inc |= len(xi) > 0
        ax.scatter(xi, yi, s=220, marker="x", c=MISSING, linewidths=2.8, zorder=5)
        ax.scatter(*pick("best"), s=1050, marker="o", facecolors="none", edgecolors=INK, linewidths=3.8, zorder=6)
    ax.set_ylabel("learning rate", color=INK, fontsize=LABEL); ax.set_xlim(min(Bs) / 1.6, max(Bs) * 1.6)
    mk = lambda kl, fill: Line2D([], [], marker=MK[kl], ls="", ms=MS[kl], mfc=KLC[kl] if fill else "none", mec=SURFACE if fill else KLC[kl], mew=1.5 if fill else 2.6)
    hs = [(mk(KLS[0], True), mk(KLS[1], True)), (mk(KLS[0], False), mk(KLS[1], False)), Line2D([], [], marker="o", ls="", ms=26.5, mfc="none", mec=INK, mew=3.8)]
    ls_ = ["reached target", "did not reach target", "fastest LR"]
    if any_inc: hs.append(Line2D([], [], marker="x", ls="", ms=15, color=MISSING, mew=2.6)); ls_.append("too early to tell")
    fig_ = ax.figure; own = fig_.get_layout_engine() is not None      # the sub-figure file (constrained layout) vs the combined figure
    def leg(fs):
        kw = dict(ncol=len(hs), frameon=False, fontsize=fs, labelcolor=INK, handletextpad=0.4, columnspacing=1.0, handlelength=1.9,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
        return fig_.legend(hs, ls_, loc="outside upper center", **kw) if own else ax.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.5, 1.0), **kw)
    fs = LEGEND - 4; l = leg(fs); fig_.canvas.draw()
    while l.get_window_extent().width > 0.98 * (fig_.bbox.width if own else ax.get_window_extent().width * 1.15) and fs > 18:   # one row, smaller type if needed
        l.remove(); fs -= 1; l = leg(fs); fig_.canvas.draw()


# ---------------- (b) fastest LR vs highest stable LR per KL
STABLE_LABEL = {"reach": "largest LR that reaches", "hold": "largest LR that holds target", "peak": "highest stable LR"}
FAIL_LABEL = {"reach": "did not reach", "hold": "did not reach / hold target", "peak": "collapsed / did not reach"}
def panel_b_band(ax):
    curves = []      # (x, y) of everything drawn, for placing the legend under the data
    for kl, nudge in zip(KLS, (-0.07, +0.07)):
        bs = np.array(sorted(set(best[kl]) & set(ceil[kl])), float) * 2.0 ** nudge
        yb = np.array([best[kl][int(round(v))] for v in bs / 2.0 ** nudge]); yc = np.array([ceil[kl][int(round(v))] for v in bs / 2.0 ** nudge])
        yf = np.array([floor[kl][int(round(v))] for v in bs / 2.0 ** nudge])
        ax.fill_between(bs, yf if a.floor else yb, yc, color=KLC[kl], alpha=0.12, lw=0)      # --floor: the whole window of LRs that reach
        ax.plot(bs, yc, ls="-", lw=4.8, color=KLC[kl], zorder=5)
        ax.plot(bs, yb, ls=(0, (5, 2.5)), lw=4.0, color=KLC[kl], alpha=0.8, zorder=4)
        if a.floor:      # smallest LR that reaches the target: the bottom of the window of LRs that work
            ax.plot(bs, yf, ls=":", lw=4.0, color=KLC[kl], alpha=0.9, zorder=4)
        curves += [(bs, yc), (bs, yb)] + ([(bs, yf)] if a.floor else [])
        ax.scatter(bs, yb, s=MS[kl] ** 2 * 1.5, marker=MK[kl], color=KLC[kl], edgecolors=SURFACE, linewidths=1.5, zorder=6)
        for x in rows:   # runs above the ceiling that did not qualify
            if x["kl"] == kl and x["out"] in ("reached_collapsed", "never") and x["lr"] > ceil[kl].get(x["B"], 0):
                ax.scatter([x["B"] * 2.0 ** nudge], [x["lr"]], s=110, marker="o", facecolors="none", edgecolors=KLC[kl], alpha=0.35, linewidths=1.8, zorder=3)
    style(ax); ax.set_ylabel("learning rate", color=INK, fontsize=LABEL); ax.set_xlim(min(Bs) / 1.9, max(Bs) * 1.9)
    lo = min(min((floor if a.floor else best)[kl].values()) for kl in KLS); ax.set_ylim(min(lo / 1.2, 9e-7), ax.get_ylim()[1])   # down to the 1e-6 tick; the legend sits in the empty lower-right corner
    hnd = [Line2D([], [], ls=(0, (5, 2.5)), lw=4.0, marker="o", ms=12, color=INK2, mec=SURFACE, label="fastest LR"),
           Line2D([], [], ls="-", lw=4.8, color=INK2, label=STABLE_LABEL[a.stable]),
           *([Line2D([], [], ls=":", lw=4.0, color=INK2, label="smallest LR that reaches" if a.stable == "reach" else "smallest LR that reaches target")] if a.floor else []),
           Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK2, alpha=0.6, mew=1.8, label=FAIL_LABEL[a.stable])]
    leg = ax.legend(handles=hnd, loc="lower right", frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.25, handlelength=1.6)      # the lower-right quadrant of the panel is empty, so the legend sits there instead of under the figure
    if a.floor:      # the floor lines run low, so lower the axis just enough that the legend's top clears every line over the legend's x span
        ax.figure.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transAxes.inverted()); x0 = ax.transData.inverted().transform(ax.transAxes.transform((bb.x0, 0)))[0]
        ystar = min(min(np.exp(np.interp(np.log(x0), np.log(cx), np.log(cy))), cy[cx >= x0].min(initial=np.inf)) for cx, cy in curves)
        f = bb.y1 + 0.03; top = ax.get_ylim()[1]       # legend top (axes fraction) plus a small gap
        ax.set_ylim(np.exp((np.log(ystar) - f * np.log(top)) / (1 - f)), top)


paper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper"); os.makedirs(paper, exist_ok=True)
# sub-figures for a LaTeX subfigure layout (a: 0.53, b: 0.46 of the line width): not cropped, same canvas height and top margin, so the
# two top-align when LaTeX sets them side by side (as for the LR split figure)
SUF = ""
for tag, fn, wfrac in (("a", panel_a, 0.53), ("b", panel_b, 0.46)):
    fig, ax = plt.subplots(figsize=(wfrac * 26.0, a.height), facecolor=SURFACE, layout="constrained"); fn(ax)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_kl_stability{SUF}_{tag}.{ext}"), dpi=170, facecolor=SURFACE)
    fig.savefig(os.path.join(paper, f"fig_kl_stability{SUF}_{tag}.pdf"), facecolor=SURFACE)
    plt.close(fig)
fig, (axl, axr) = plt.subplots(1, 2, figsize=(26.0, a.height), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.22, wspace=0.22)
panel_a(axl); panel_b(axr)
if a.panel_b != "band":      # the shared KL legend, centred above both panels
    fig.legend(handles=KL_HANDLES(), loc="lower center", bbox_to_anchor=(0.5, 0.86), ncol=2, frameon=False, fontsize=LEGEND - 4, labelcolor=INK,
               handletextpad=0.5, columnspacing=1.4, handlelength=2.2)
for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_kl_stability{SUF}.{ext}"), dpi=170, facecolor=SURFACE, bbox_inches="tight")
fig.savefig(os.path.join(paper, f"fig_kl_stability{SUF}.pdf"), facecolor=SURFACE, bbox_inches="tight")   # paper copy, next to the other paper-ready PDFs
plt.close(fig)
# the shared KL legend as its own file, for a LaTeX layout that sets it above the two sub-figures
fig = plt.figure(figsize=(0.4 * 26.0, 1.0), facecolor=SURFACE)
fig.legend(handles=KL_HANDLES(), loc="center", ncol=2, frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.5, columnspacing=1.4, handlelength=2.2)
for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"fig_kl_stability{SUF}_legend.{ext}"), dpi=170, facecolor=SURFACE, bbox_inches="tight")
fig.savefig(os.path.join(paper, f"fig_kl_stability{SUF}_legend.pdf"), facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
print("wrote", os.path.join(a.out, f"fig_kl_stability{SUF}.png"), f"and paper/fig_kl_stability{SUF}.pdf (+ _a, _b, _legend)")
for kl in KLS:
    print(KLN[kl], f"highest stable LR (rule: {a.stable}):", {b: f"{ceil[kl][b]:g}" for b in sorted(ceil[kl])})
    if a.floor:   # is the floor measured (a lower LR was tried and did not reach) or just the lowest LR tried?
        tried = {}
        for x in rows:
            if x["kl"] == kl: tried.setdefault(x["B"], []).append(x["lr"])
        print(KLN[kl], "smallest LR that reaches (* = lowest LR tried, so the true floor may be lower):",
              {b: f"{floor[kl][b]:g}" + ("*" if floor[kl][b] <= min(tried[b]) else "") for b in sorted(floor[kl])})
