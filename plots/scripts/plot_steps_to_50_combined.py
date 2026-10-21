"""Two LR grids side by side, sharing the learning-rate axis, the colour scale and the legend:
  left   the batch sweep   x = batch size B at K rollouts per prompt (default 16)
  right  the rollout sweep x = rollouts per prompt K at batch B (default 1024)
Both at KL 1e-3, plain GRPO. Paper styling: no title, no step labels, no star for the fastest cell, no pre-/post-fix marking. Same rules as plot_steps_to_50.py: runs with identical (K, B, LR) collapse to the fastest;
a run that never reached the target is "too early to tell" when it logged fewer steps than the fastest crossing at its
x (capped at SHORT_STEPS); the fastest LR per x is ringed; "too early to tell" runs are drawn only in columns whose fastest LR is not yet bracketed by a
definite worse LR on both sides; the batch sweep gets the
power-law fit of the fastest LR against B (up to the plateau), the rollout sweep is not fitted.

Usage: python plot_steps_to_50_combined.py csv/steps_to_50_kl.csv png/steps_to_50_combined.png [--n 16] [--bsz 1024] [--pct 50] [--lr-max 1e-4] [--lr-min 5e-7] [--fixed-only] [--batch-only] [--n2 64] [--split] [--nstar csv/steps_vs_seqs_b1024_fit_fit_table.csv]
  --nstar marks each sweep's fitted critical batch (B* for a prompt sweep, K* for a rollout sweep) from that Figure 1 fit table.
  --split writes <stem>_a (the prompt sweeps), <stem>_b (the rollout sweep with the colour bar) and <stem>_legend (one-row strip)
  as separate PNG/PDF files sized 0.655 : 0.33 : 1 of one line width, for a LaTeX subfigure layout with a shared legend.
  --n2 K adds a third panel, the batch sweep at that K, drawn from fixed-code runs only (its pre-fix runs stepped at ~0.46x nominal LR).
  --batch-only draws just the batch-sweep panel (same styling, legend and colour scale).
  --fixed-only keeps only runs on the fixed loss normalisation (for K = 64, whose few pre-fix runs stepped at ~0.46x nominal LR).
  --fig1-colors colours each panel in its Figure 1 series colour (K = 16 blue, K = 64 orange, B = 128 green, B = 1024 purple):
  the reached runs on a light-to-dark ramp of that hue (darker = fewer steps), the fit and the critical-batch line in the hue;
  the shared colour bar becomes a neutral grey lightness scale.
  --no-nstar-n2 leaves the critical-batch line off the --n2 panel.
  --no-fit-n2 leaves the power-law fit and its exponent off the --n2 panel (its LRs were set by the K = 16 rule, so the fit is circular).
  --share-b gives the prompt-sweep panels one common batch-size axis (and equal widths).
  --no-early does not draw the "too early to tell" runs (nor their legend entry).
  --lr-max / --lr-min drop runs outside that learning-rate range (the paper figure uses 5e-7 to 1e-4: nothing outside it ever
  crossed, those runs only add headroom).
Writes the PNG, paper/<stem>.pdf and csv/<stem>_table.csv (both panels' collapsed grids).
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue_rev", RAMP[::-1])      # dark = fewest steps
TICK, LABEL, TITLE, NOTE, LEGEND, CBAR = 28, 31, 32, 29, 34, 31     # type sizes (points at the rendered figure size)
MISSING = "#8f8e8a"          # darker grey than the single-panel grids, for contrast against the surface
STARC = "#6a3d9a"            # the fitted critical batch N*: purple, distinct from everything else on the panel
FITC = "#d66300"             # the power-law fit: deep orange, distinct from the blue colour scale and the grey marks
SHORT_STEPS = 200
# Figure 1 series colours (plot_steps_vs_seqs_paper.py): K = 16 tab:blue, K = 64 deep orange, B = 128 tab:green, B = 1024 tab:purple
FIG1_HUE = {("bsz", 16): "#1f77b4", ("bsz", 64): "#d66300", ("n", 128): "#2ca02c", ("n", 1024): "#9467bd"}


def hue_ramp(hex_):
    """light (slow) -> dark (fast) ramp of one hue, reversed like CMAP so that dark = fewest steps"""
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(hex_)); white, black = np.ones(3), np.zeros(3)
    stops = [0.80 * white + 0.20 * c, 0.45 * white + 0.55 * c, c, 0.78 * c + 0.22 * black, 0.55 * c + 0.45 * black]
    return LinearSegmentedColormap.from_list("hue_rev", stops[::-1])


GREY_RAMP = LinearSegmentedColormap.from_list("grey_rev", ["#1e1e1e", "#5b5b5b", "#9a9a9a", "#d6d6d6"])


def opt(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
csv_in, png_out = args[0], args[1]
N_ROLL, BSZ, PCT = opt("--n", 16, int), opt("--bsz", 1024, int), opt("--pct", 50, int)
LR_MAX = opt("--lr-max", None, float)      # drop runs above this learning rate (they only add headroom when none of them crossed)
LR_MIN = opt("--lr-min", None, float)      # likewise below it
BATCH_ONLY = "--batch-only" in sys.argv    # one panel: just the batch sweep, same styling
SPLIT = "--split" in sys.argv              # write (a) the prompt sweeps, (b) the rollout sweep with the colour bar, and a legend strip as separate files
NSTAR = opt("--nstar", None, str)          # Figure 1 fit table (csv/steps_vs_seqs_*fit_fit_table.csv): mark each sweep's fitted critical batch N* on its panel
N2 = opt("--n2", None, int)                # third panel: a second batch sweep at this K (fixed-code runs only; its pre-fix runs stepped at a fraction of the nominal LR)
FIG1 = "--fig1-colors" in sys.argv          # per-panel Figure 1 hues instead of one blue scale
NO_NSTAR_N2 = "--no-nstar-n2" in sys.argv      # no critical-batch line on the second batch sweep
NO_FIT_N2 = "--no-fit-n2" in sys.argv          # no power-law fit on the second batch sweep
NO_EARLY = "--no-early" in sys.argv
SHARE_B = "--share-b" in sys.argv              # the prompt-sweep panels use one common batch-size axis            # do not draw the "too early to tell" runs
# --split is the paper layout: canvas and sizes from paperstyle.py (the shared figure style; Figure 1 is the reference)
if SPLIT:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paperstyle as PS
    TICK, LABEL, TITLE, NOTE, LEGEND, CBAR = PS.FS["tick"], PS.FS["label"], PS.FS["title"], PS.FS["annot"], PS.FS["legend"], PS.FS["label"]
    SZ = dict(hit=95, miss=70, short=65, ring=250, ring_lw=2.2, edge_lw=1.8, fit_lw=PS.LW["line"], plateau_lw=3.2, star_lw=PS.LW["crit"],
              leg_ms=10, leg_ring=13, strip_h=1.2, W=PS.TEXTWIDTH_IN * PS.K_CANVAS, H=4.9)
else:
    SZ = dict(hit=260, miss=190, short=170, ring=720, ring_lw=3.4, edge_lw=2.6, fit_lw=4.6, plateau_lw=3.6, star_lw=3.6,
              leg_ms=16, leg_ring=22, strip_h=2.3, W=26.0, H=7.4)
FIXED_ONLY = "--fixed-only" in sys.argv    # drop pre-fix runs (before 2026-09-11 they stepped at a fraction of the nominal LR; see the loss-scaling note)

df = pd.read_csv(csv_in)
COL = f"steps_to_{PCT}_interp" if f"steps_to_{PCT}_interp" in df else f"steps_to_{PCT}"
df = df.dropna(subset=["bsz", "lr"])
if "kl" in df:
    df = df[df["kl"].round(6) == 1e-3]
df["fixed"] = df["name"].str.contains("updated_scaling", na=False)
if LR_MAX is not None:
    df = df[df["lr"] <= LR_MAX * 1.001]
if LR_MIN is not None:
    df = df[df["lr"] >= LR_MIN / 1.001]
if FIXED_ONLY:
    df = df[df["fixed"]]
# a run validated every step ("_dense") that crossed replaces the coarsely validated runs of its (n, B, LR) cell (same rule as Figure 1)
_cell = ["n", "bsz", "lr"]
_isd = df["name"].str.contains("_dense", na=False)
_dkey = set(df.loc[_isd & df[COL].notna(), _cell].astype(str).agg("|".join, axis=1))
df = df[~(df[_cell].astype(str).agg("|".join, axis=1).isin(_dkey) & ~_isd)]


def collapse(sub):
    """one row per (K, B, LR): the fastest run, with the count, the longest run and whether the fastest is post-fix"""
    sub = sub.sort_values(COL, na_position="last")
    g = sub.groupby(["n", "bsz", "lr"], as_index=False).agg(steps=(COL, "min"), n_runs=("id", "count"), longest=("last_val_step", "max"),
                                                            max_val=("max_val", "max"), fixed=("fixed", "first"))
    g["n"] = g["n"].astype(int); g["bsz"] = g["bsz"].astype(int)
    return g


batch = collapse(df[df["n"] == N_ROLL])          # x = bsz
roll = collapse(df[df["bsz"] == BSZ])            # x = n
batch2 = collapse(df[(df["n"] == N2) & df["fixed"]]) if N2 else batch.iloc[0:0]
both = pd.concat([batch.assign(panel="batch_sweep"), roll.assign(panel="rollout_sweep"), batch2.assign(panel="batch_sweep_2")])
both.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)
hit = both.dropna(subset=["steps"])
norm = LogNorm(vmin=hit["steps"].min(), vmax=hit["steps"].max())     # one colour scale for both panels


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(not SPLIT); ax.spines[s].set_color(GRID)      # paper layout: no frame, as in Figure 1
    ax.tick_params(colors=INK, labelsize=TICK, length=0)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel(ax, sub, xcol, xlabel, fit, star=None, cmap=CMAP, fitc=FITC, starc=STARC):
    ok = sub.dropna(subset=["steps"])
    miss = sub[sub["steps"].isna()]
    ref = miss[xcol].map(ok.groupby(xcol)["steps"].min()).fillna(SHORT_STEPS).clip(upper=SHORT_STEPS)
    short = miss[miss["longest"].fillna(0) < ref]
    miss = miss[miss["longest"].fillna(0) >= ref]
    # "too early to tell" runs matter only where the fastest LR is not yet bracketed: a column whose best LR has a
    # worse LR (slower crossing or never) immediately below and above it is settled, so its inconclusive runs are not drawn
    settled = set()
    for x, col in sub.groupby(xcol):
        okx = col.dropna(subset=["steps"])
        if okx.empty:
            continue
        lrs = sorted(col["lr"].unique()); i = lrs.index(okx.loc[okx["steps"].idxmin(), "lr"])
        def definite(j):
            if j < 0 or j >= len(lrs):
                return False
            r = col[col["lr"] == lrs[j]]
            return r["steps"].notna().any() or (r["longest"].fillna(0) >= min(okx["steps"].min(), SHORT_STEPS)).any()
        if definite(i - 1) and definite(i + 1):
            settled.add(x)
    short = short[~short[xcol].isin(settled)]
    ax.scatter(miss[xcol], miss["lr"], s=SZ["miss"], facecolors="none", edgecolors=MISSING, linewidths=SZ["edge_lw"], zorder=2)
    if NO_EARLY:
        short = short.iloc[0:0]
    ax.scatter(short[xcol], short["lr"], s=SZ["short"], marker="x", c=MISSING, linewidths=SZ["edge_lw"], zorder=2)
    ax.scatter(ok[xcol], ok["lr"], s=SZ["hit"], c=ok["steps"], cmap=cmap, norm=norm, edgecolors=SURFACE, linewidths=1.5, zorder=3)

    best = ok.loc[ok.groupby(xcol)["steps"].idxmin()].sort_values(xcol)
    ax.scatter(best[xcol], best["lr"], s=SZ["ring"], facecolors="none", edgecolors=INK, linewidths=SZ["ring_lw"], zorder=4)

    note = None; plateau = False
    if fit and len(best) >= 3:
        lrs = best["lr"].to_numpy(); k = len(lrs)
        while k >= 2 and np.isclose(lrs[k - 2], lrs[-1]):
            k -= 1
        plateau = len(lrs) - k >= 2
        fitpts = best.iloc[:k] if plateau else best
        lx, ly = np.log(fitpts[xcol].astype(float)), np.log(fitpts["lr"].astype(float))
        slope, icpt = np.polyfit(lx, ly, 1)
        resid = ly - (slope * lx + icpt); se = np.sqrt(resid.var(ddof=2) / ((lx - lx.mean()) ** 2).sum()) if len(fitpts) > 2 else float("nan")
        xs_fit = np.geomspace(fitpts[xcol].min(), fitpts[xcol].max(), 50)
        ax.plot(xs_fit, np.exp(icpt) * xs_fit ** slope, ls="-", lw=SZ["fit_lw"], color=fitc, alpha=0.95, zorder=2.6)
        if plateau:
            ax.plot([fitpts[xcol].max(), best[xcol].max()], [lrs[-1]] * 2, ls=":", lw=SZ["plateau_lw"], color=fitc, alpha=0.95, zorder=2.6)
            note = f"$\\eta^* \\propto B^{{{slope:.2f}}}$ (±{se:.2f})"
        else:
            note = f"$\\eta^* \\propto B^{{{slope:.2f}}}$ (±{se:.2f})"
        ax.text(*((0.97, 0.05) if SPLIT else (0.03, 0.93)), note, transform=ax.transAxes, ha="right" if SPLIT else "left", va="bottom" if SPLIT else "top", fontsize=NOTE, color=fitc, zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))   # top-left corner
    gb = ok[ok["steps"] == ok["steps"].min()]

    if star is not None:          # the sweep's fitted critical batch from the steps-vs-sequences fit, in this panel's x units (legend names it)
        ax.axvline(star, ls=(0, (5, 2.5)), lw=SZ["star_lw"], color=starc, alpha=0.9, zorder=2.3)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    drawn = pd.concat([ok, miss, short])
    xs = sorted(sub[xcol].unique())
    fmt = lambda x: f"{int(x) // 1024}k" if x >= 1024 else str(int(x))         # 1k, 2k, ...
    if len(xs) > (6 if SPLIT else 8):        # a long batch-size axis: label every other doubling (even powers of two: 4, 16, 64, 256, 1k, 4k, 16k), tick the rest
        major = [x for x in xs if int(round(np.log2(x))) % 2 == 0]; minor = [x for x in xs if x not in major]
        ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major])
        ax.set_xticks(minor, minor=True); ax.set_xticklabels([""] * len(minor), minor=True); ax.tick_params(axis="x", which="minor", length=0)
    else:
        ax.set_xticks(xs); ax.set_xticklabels([fmt(x) for x in xs])
    ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
    if SPLIT:       # room for the large rings at both ends of the axis
        ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    style(ax)
    return best, gb, note, drawn, plateau


# panels in order: the batch sweeps (K = N_ROLL, then K = N2 if given), then the rollout sweep at B = BSZ (unless --batch-only)
specs = [dict(data=batch, xcol="bsz", xlabel=(f"batch size $B$  ($K={N_ROLL}$)" if SPLIT else "batch size B"), title=(f"prompts, K = {N_ROLL}" if SPLIT else f"scaling prompts, fixed K = {N_ROLL}"), fit=True)]
if N2:
    specs.append(dict(data=batch2, xcol="bsz", xlabel=(f"batch size $B$  ($K={N2}$)" if SPLIT else "batch size B"), title=(f"prompts, K = {N2}" if SPLIT else f"scaling prompts, fixed K = {N2}"), fit=not NO_FIT_N2))
if not BATCH_ONLY:
    specs.append(dict(data=roll, xcol="n", xlabel=(f"rollouts $K$ ($B={BSZ}$)" if SPLIT else "rollouts per prompt K"), title=(f"rollouts, B = {BSZ}" if SPLIT else f"scaling rollouts, fixed B = {BSZ}"), fit=False))
if NSTAR:
    ft = pd.read_csv(NSTAR).set_index("series")["star"]
    for sp in specs:
        key = f"more prompts, K = {N_ROLL if sp is specs[0] else N2}" if sp["xcol"] == "bsz" else f"more rollouts, B = {BSZ}"
        sp["star"] = float(ft[key]) if key in ft else None
        if NO_NSTAR_N2 and N2 and sp is not specs[0] and sp["xcol"] == "bsz":
            sp["star"] = None; continue
        if sp["star"] is None:
            print(f"note: no N* for '{key}' in {NSTAR}")
widths = [max(sp["data"][sp["xcol"]].nunique(), 1) for sp in specs]
H = SZ["H"]
if SPLIT:
    # (a) the batch sweeps side by side, (b) the rollout sweep; widths 0.655 : 0.33 of the same line, so the type matches in the paper
    W = SZ["W"]
    bspecs, rspecs = [sp for sp in specs if sp["xcol"] == "bsz"], [sp for sp in specs if sp["xcol"] == "n"]
    figA, gA = plt.subplots(1, len(bspecs), figsize=(0.655 * W, H), facecolor=SURFACE, sharey=True, layout="constrained",
                            gridspec_kw=dict(width_ratios=[1] * len(bspecs) if SHARE_B else [max(sp["data"]["bsz"].nunique(), 1) for sp in bspecs]), squeeze=False)
    figA.get_layout_engine().set(wspace=0.04)
    figB, gB = plt.subplots(1, 1, figsize=(0.33 * W, H), facecolor=SURFACE, layout="constrained", squeeze=False)
    axes = list(gA[0]) + list(gB[0]); specs = bspecs + rspecs; fig = figA
else:
    fig_w = {1: 14.0, 2: 20.0, 3: 26.0}[len(specs)]
    GAP = 1.4 if (N2 and not BATCH_ONLY) else 0          # several batch sweeps: an empty spacer column sets the rollout sweep apart
    cols = widths[:-1] + ([GAP] if GAP else []) + widths[-1:]
    LEG_H = 2.4 if len(specs) == 1 else 1.7                # room for the in-figure legend below the axes (one column / two or three)
    fig, grid = plt.subplots(1, len(cols), figsize=(fig_w, H + LEG_H), facecolor=SURFACE, sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=cols), squeeze=False)
    grid = list(grid[0])
    if GAP:
        grid[-2].set_visible(False); axes = grid[:-2] + grid[-1:]
    else:
        axes = grid
    if len(axes) > 1:
        fig.get_layout_engine().set(wspace=0.04 if len(axes) < 3 else 0.06)
results = []
for ax, sp in zip(axes, specs):
    kw = {}
    if FIG1:
        h = FIG1_HUE.get(("bsz", N_ROLL if sp is specs[0] else N2) if sp["xcol"] == "bsz" else ("n", BSZ), INK2)
        kw = dict(cmap=hue_ramp(h), fitc=h, starc=h)
    best, gb, note, drawn, plateau = panel(ax, sp["data"], sp["xcol"], sp["xlabel"], fit=sp["fit"], star=sp.get("star"), **kw)
    if not SPLIT:   # paper layout: no titles on top; the sweep is in the x-label and the sub-caption sits under the panels
        ax.set_title(sp["title"], loc="left", fontsize=TITLE if len(specs) < 3 else TITLE - 2, color=INK, pad=4)
    results.append(dict(spec=sp, best=best, note=note, drawn=drawn, plateau=plateau))
if SPLIT:        # sub-figure captions inside the figure, under the panels (same height in both files, so they line up in LaTeX)
    figA.supxlabel("(a) Learning rate vs. $B$.", fontsize=LABEL)
    figB.supxlabel("(b) Learning rate vs. $K$.", fontsize=LABEL)
lr_all = pd.concat([r["drawn"] for r in results])["lr"]          # shared y-limits from what the panels actually draw
for ax in axes:
    ax.set_ylim(lr_all.min() / 1.7, lr_all.max() * 1.7)
if SHARE_B:     # the prompt-sweep panels share one batch-size axis (the K = 16 range, 4 ... 16k), so they line up
    bax = [ax for ax, sp in zip(axes, specs) if sp["xcol"] == "bsz"]
    lo = min(sp["data"]["bsz"].min() for sp in specs if sp["xcol"] == "bsz"); hi = max(sp["data"]["bsz"].max() for sp in specs if sp["xcol"] == "bsz")
    allb = [2 ** i for i in range(int(np.log2(lo)), int(np.log2(hi)) + 1)]
    major = [b for b in allb if int(round(np.log2(b))) % 4 == 2]      # 4, 64, 1k, 16k: room for the larger type
    for ax in bax:
        ax.set_xlim(lo / 1.6, hi * 1.6)
        ax.set_xticks(major); ax.set_xticklabels([f"{b // 1024}k" if b >= 1024 else str(b) for b in major])
        ax.set_xticks([b for b in allb if b not in major], minor=True); ax.set_xticklabels([], minor=True)
axes[0].set_ylabel("learning rate", color=INK, fontsize=LABEL)
plateau_b = any(r["plateau"] for r in results)

sm = plt.cm.ScalarMappable(norm=norm, cmap=GREY_RAMP if FIG1 else CMAP); sm.set_array([])
if SPLIT:
    cb = figB.colorbar(sm, ax=list(gB[0]), fraction=0.06, pad=0.02)       # the colour bar rides with panel (b)
else:
    cb = fig.colorbar(sm, ax=axes, fraction={1: 0.04, 2: 0.025, 3: 0.02}[len(axes)], pad=0.015)
cb.set_label("steps to target", color=INK, fontsize=CBAR)
cb.ax.tick_params(colors=INK, labelsize=TICK - 1, length=0); cb.outline.set_visible(False); cb.ax.invert_yaxis()

legend = [
    Line2D([], [], marker="o", ls="", ms=SZ["leg_ms"], mfc="#5b5b5b" if FIG1 else RAMP[4], mec=SURFACE, label="reached target"),
    Line2D([], [], marker="o", ls="", ms=SZ["leg_ms"], mfc="none", mec=MISSING, mew=SZ["edge_lw"], label="never reached target"),
] + ([] if NO_EARLY else [Line2D([], [], marker="x", ls="", ms=SZ["leg_ms"], color=MISSING, mew=SZ["edge_lw"], label="too early to tell")]) + [
    Line2D([], [], marker="o", ls="", ms=SZ["leg_ring"], mfc="none", mec=INK, mew=SZ["ring_lw"], label="fastest LR at each x"),
    Line2D([], [], ls="-", lw=SZ["fit_lw"], color=INK2 if FIG1 else FITC, label="power-law fit (dotted: plateau)" if plateau_b else "power-law fit"),
]
if NSTAR and any(sp.get("star") is not None for sp in specs):
    legend.append(Line2D([], [], ls=(0, (5, 2.5)), lw=SZ["star_lw"], color=INK2 if FIG1 else STARC,
                         label="fitted critical batch (B* or K*)" if FIG1 else "critical batch (B* or K*)"))
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
if SPLIT:
    stem = png_out.replace(".png", "")
    # not cropped: both files keep the full H-inch canvas with the same top margin, so the titles sit at the same height
    # when LaTeX top-aligns the two subfigures (a tight crop differs between the two and shifts one title)
    for tag, f in (("a", figA), ("b", figB)):
        f.savefig(f"{stem}_{tag}.png", dpi=170, facecolor=SURFACE)
        f.savefig(os.path.join(pdf_dir, os.path.basename(stem) + f"_{tag}.pdf"), facecolor=SURFACE)
        print("wrote", f"{stem}_{tag}.png")
    # legend strip at the full line width (W), so its type comes out at the same size as the panels; not cropped, to keep that scale
    figL = plt.figure(figsize=(W, SZ["strip_h"]), facecolor=SURFACE)
    # two rows, each centred on its own (one legend per row), so a short second row does not leave the strip lopsided
    half = (len(legend) + 1) // 2
    for rows_, y in ((legend[:half], 0.73), (legend[half:], 0.27)):
        figL.legend(handles=rows_, loc="center", bbox_to_anchor=(0.5, y), ncol=len(rows_), frameon=False, fontsize=LEGEND, labelcolor=INK,
                    handletextpad=0.6, columnspacing=2.0, borderaxespad=0.0)
    figL.savefig(f"{stem}_legend.png", dpi=170, facecolor=SURFACE)
    figL.savefig(os.path.join(pdf_dir, os.path.basename(stem) + "_legend.pdf"), facecolor=SURFACE)
    print("wrote", f"{stem}_legend.png")
else:
    fig.legend(handles=legend, loc="outside lower center", ncol={1: 2, 2: 3, 3: 3}[len(axes)], frameon=False, fontsize=LEGEND, labelcolor=INK,
               handletextpad=0.6, columnspacing=1.6)     # constrained layout packs it directly under the axes
    fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), bbox_inches="tight", facecolor=SURFACE)
    print("wrote", png_out)
for r in results:
    sp = r["spec"]
    print(f"\n{sp['title']}" + (f"   {r['note']}".replace("\n", " ") if r["note"] else ""))
    print(r["best"][[sp["xcol"], "lr", "steps", "n_runs"]].to_string(index=False))
