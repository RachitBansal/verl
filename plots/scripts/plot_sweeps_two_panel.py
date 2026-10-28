"""Steps to target under each kind of batch scaling, all four sweeps in two panels (the companion of Figure 1).

  (a) scaling prompts: steps to the target against the prompt batch size B, for the K = 16 and K = 64 sweeps;
  (b) scaling rollouts: steps against the rollouts per prompt K, for the B = 128 and B = 1024 sweeps.
Each sweep is drawn in its Figure 1 colour with its own fit S = S_min (1 + N*/N) (read from the Figure 1 fit table, not refitted)
and its critical point, B* = N*/K or K* = N*/B (with --lin 0.2: where linear scaling ends, 0.2 N*), as a light dashed vertical labelled at the top. Both panels share the y range.
Data and admission rules are those of plot_steps_vs_seqs_paper.py (KL 1e-3; fastest LR per cell; pre-fix runs admitted at
K = 16 and for K <= 16 at B = 128; K = 64 and B = 1024 fixed code only, dense reruns replacing coarse runs at K = 64).

Usage: python plot_sweeps_two_panel.py csv/steps_to_50_seqs.csv png/sweeps_two_panel.png [--fit-table csv/steps_vs_seqs_b1024_fit_fit_table.csv]
Writes the PNG, paper/<stem>.pdf, the sub-figure files <stem>_a / <stem>_b (0.495 of the line width each, uncropped, equal height),
<stem>_legend (a strip for the LaTeX subfigure layout) and csv/<stem>_table.csv.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Paper scale (shared style with the iclr_figs figures and paperstyle.py): this script lays figures out on a 26-inch "line width"
# with 28-34 pt type; drawing it on a 19-inch canvas instead (x 19/26) prints the type at ~8-10 pt and lines at ~1.4 pt when the
# figure is scaled to the 5.5-inch text width, the same as the other paper figures.
PAPER_SCALE = 19.0 / 26.0
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
C = {"n16": "#1f77b4", "n64": "#d66300", "b128": "#2ca02c", "b1024": "#9467bd"}       # Figure 1 colours
MARKER_SHRINK = 0.72      # smaller markers (curves keep their width) so neighbouring points do not overlap
MK = {"n16": ("o", 15.5), "n64": ("h", 16.5), "b128": ("p", 17), "b1024": ("v", 17)}
TICK, LABEL, LEGEND = 28, 31, 27      # with PAPER_SCALE: ~8 / 9 / 7.8 pt printed, as paperstyle.py
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}
FIXED_ONLY_N = {32, 64, 128}


def opt(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
# --lin FRAC (as in plot_fig1_one_axis.py): mark where linear batch-to-steps scaling ends, N_lin = FRAC N* (the fit is (1 + FRAC) x
# perfect scaling there; FRAC = 0.2: 20% above), with an open circle on the fit, instead of the fitted N* itself
LIN = opt("--lin", None, float)


args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
csv_in, png_out = args[0], args[1]
FIT_TABLE = opt("--fit-table", "csv/steps_vs_seqs_b1024_fit_fit_table.csv", str)

df = pd.read_csv(csv_in)
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df = df[np.isclose(df["kl"].astype(float), 1e-3)].dropna(subset=["seqs", "lr"]).copy()
df = df[~df["downsample"].astype(bool)]
df = df[(df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[COL].notna()]
fixed = df["fixed"].astype(bool)


def fastest_per(src, key):
    rows = []
    for k, s in src.groupby(key):
        s = s.sort_values([COL, "last_val_step"], na_position="last", ascending=[True, False]); p = s.iloc[0]
        rows.append(dict(x=int(k), seqs=int(p["seqs"]), lr=p["lr"], steps=p[COL], run=p["name"]))
    return pd.DataFrame(rows).sort_values("x").dropna(subset=["steps"])


b64_src = df[(df["n"] == 64) & fixed]
dense_b = set(b64_src.loc[b64_src["name"].str.contains("_dense"), "bsz"])
b64_src = b64_src[~(b64_src["bsz"].isin(dense_b) & ~b64_src["name"].str.contains("_dense"))]
SWEEPS = {   # key -> (points, per-unit, fit-table series, panel, x label unit)
    "n16":   (fastest_per(df[df["n"] == 16], "bsz"), 16, "more prompts, K = 16", "a"),
    "n64":   (fastest_per(b64_src, "bsz"), 64, "more prompts, K = 64", "a"),
    "b128":  (fastest_per(df[(df["bsz"] == 128) & (df["n"].isin(VALID_N_PREFIX) | (fixed & df["n"].isin(FIXED_ONLY_N)))], "n"), 128, "more rollouts, B = 128", "b"),
    "b1024": (fastest_per(df[(df["bsz"] == 1024) & fixed], "n"), 1024, "more rollouts, B = 1024", "b"),
}
NAME = {"n16": "prompts, K = 16", "n64": "prompts, K = 64", "b128": "rollouts, B = 128", "b1024": "rollouts, B = 1024"}
fits = pd.read_csv(FIT_TABLE).set_index("series")
pd.concat([v[0].assign(sweep=k) for k, v in SWEEPS.items()]).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

allsteps = pd.concat([v[0]["steps"] for v in SWEEPS.values()])
ylo, yhi = allsteps.min() / 1.6, allsteps.max() * 1.6
fmt = lambda v: f"{int(v) // 1024}k" if v >= 1024 else str(int(v))


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(ylo, yhi)


def panel(ax, keys, xlabel, star_sym, side="left"):
    """the sweeps of one panel: points, fit, perfect-scaling reference and critical point for each, labels at the top"""
    xs_all = set()
    for i, key in enumerate(keys):
        pts, per, series, _ = SWEEPS[key]; col = C[key]; mk, ms = MK[key]
        f = fits.loc[series]; smin, nstar = float(f["s_min"]), float(f["n_star_seqs"])
        nm = LIN * nstar if LIN else nstar; star = nm / per
        xs = np.geomspace(pts["x"].min() / 1.5, pts["x"].max() * 1.5, 200)
        ax.plot(xs, smin * (1 + nstar / (per * xs)), ls="-", lw=4.8, color=col, zorder=3)
        ax.plot(xs, smin * nstar / (per * xs), ls=":", lw=2.6, color=col, alpha=0.6, zorder=2.5)       # perfect scaling, per sweep
        ax.scatter(pts["x"], pts["steps"], s=(MARKER_SHRINK * ms) ** 2 * 1.5, marker=mk, color=col, edgecolors=SURFACE, linewidths=1.2, zorder=6)
        ax.axvline(star, ls=(0, (3, 3)), lw=2.6, color=col, alpha=0.55, zorder=2.3)
        if LIN:     # the point on the fit where linear scaling ends
            ax.plot([star], [smin * (1 + nstar / nm)], marker="o", ms=15, mfc=SURFACE, mec=col, mew=3.0, ls="", zorder=8)
        txt = f"{star / 1000:.1f}k" if star >= 1000 else f"{star:.0f}"
        # labels at the top, each to the left of its own line, the second sweep's one row lower so the two never collide
        ax.annotate(f"{star_sym} $\\approx$ {txt}", (star, 0.96 - 0.10 * i), xycoords=("data", "axes fraction"), xytext=(-10 if side == "left" else 10, 0), textcoords="offset points",
                    ha="right" if side == "left" else "left", va="top", fontsize=LEGEND, color=col, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=1.0))
        xs_all |= set(pts["x"])
    style(ax)
    xs_t = sorted(xs_all); major = [x for x in xs_t if int(round(np.log2(x))) % 2 == 0] if len(xs_t) > 8 else xs_t
    ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major])
    minor = [x for x in xs_t if x not in major]
    ax.set_xticks(minor, minor=True); ax.set_xticklabels([""] * len(minor), minor=True); ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(min(xs_t) / 1.9, max(xs_t) * 1.9)
    ax.set_xlabel(xlabel, color=INK, fontsize=LABEL); ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)


def draw_a(ax): panel(ax, ["n16", "n64"], "prompts B", "$B$" if LIN else "$B^*$")
def draw_b(ax): panel(ax, ["b128", "b1024"], "rollouts K", "$K$" if LIN else "$K^*$", side="right" if LIN else "left")   # --lin: the K lines sit where the curves are high; labels to the right


# three columns, filled column by column: the two prompt sweeps | the two rollout sweeps | fit, marker, perfect scaling (a four-column
# layout is wider than the line once the marker label is "linear scaling ends (20% above perfect)")
BLANK = Line2D([], [], ls="", color=SURFACE)
sw = [Line2D([], [], marker=MK[k][0], ls="", ms=MARKER_SHRINK * MK[k][1], color=C[k], mec=SURFACE) for k in SWEEPS]
hnd = sw[:2] + [BLANK] + sw[2:] + [BLANK] + \
      [Line2D([], [], ls="-", lw=4.8, color=INK2),
       Line2D([], [], ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8, marker="o" if LIN else None, ms=12, mfc=SURFACE, mec=INK2, mew=2.4),
       Line2D([], [], ls=":", lw=2.6, color=INK2, alpha=0.8)]
names = [NAME[k] for k in SWEEPS]
lab = names[:2] + [""] + names[2:] + [""] + ["fit $S = S_\\min(1 + N^*/N)$", f"linear scaling ends (${LIN:g} \\times N^*$)" if LIN else "critical batch $N^*$",
                                             "perfect scaling"]


def check_inside(fig, leg, what):
    """fail loudly if a legend runs off its canvas (a fixed-size legend strip clips it silently)"""
    fig.canvas.draw(); bb = leg.get_window_extent(); fb = fig.bbox
    if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
        raise SystemExit(f"{what}: legend {bb.bounds} does not fit in the canvas {fb.bounds}")
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
H = 7.4
fig, (axa, axb) = plt.subplots(1, 2, figsize=(PAPER_SCALE * (26.0), PAPER_SCALE * (H + 3.6)), facecolor=SURFACE)
fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.37, wspace=0.26)
draw_a(axa); draw_b(axb)
leg = fig.legend(hnd, lab, loc="lower center", ncol=3, frameon=False, fontsize=LEGEND, labelcolor=INK, handletextpad=0.6, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
fig.savefig(png_out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), facecolor=SURFACE, bbox_inches="tight")
stem = png_out.replace(".png", "")
for tag, fn in (("a", draw_a), ("b", draw_b)):
    f_, a_ = plt.subplots(figsize=(PAPER_SCALE * (0.495 * 26.0), PAPER_SCALE * (H)), facecolor=SURFACE, layout="constrained"); fn(a_)
    f_.savefig(f"{stem}_{tag}.png", dpi=170, facecolor=SURFACE)
    f_.savefig(os.path.join(pdf_dir, os.path.basename(stem) + f"_{tag}.pdf"), facecolor=SURFACE); plt.close(f_)
fL = plt.figure(figsize=(PAPER_SCALE * (26.0), PAPER_SCALE * (3.4)), facecolor=SURFACE)
legL = fL.legend(hnd, lab, loc="center", ncol=3, frameon=False, fontsize=LEGEND, labelcolor=INK, handletextpad=0.6, columnspacing=1.2)
check_inside(fL, legL, "legend strip")
fL.savefig(f"{stem}_legend.png", dpi=170, facecolor=SURFACE); fL.savefig(os.path.join(pdf_dir, os.path.basename(stem) + "_legend.pdf"), facecolor=SURFACE); plt.close(fL)
print("wrote", png_out, "with _a / _b / _legend (PNG + paper PDFs)")
for k, (pts, per, series, _) in SWEEPS.items():
    f = fits.loc[series]; print(f"{NAME[k]:<20} S_min {float(f['s_min']):6.2f}  N* {float(f['n_star_seqs']):>9,.0f} seqs -> {float(f['n_star_seqs']) / per:>7,.0f} {'prompts' if per in (16, 64) else 'rollouts'}; {len(pts)} points")
