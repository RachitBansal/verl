"""Figure 1 (option B): scaling prompts versus scaling rollouts, two panels at the same scale.

  (a) steps to the target against the prompt batch size B at fixed K = 16, with the Figure 1 fit S = S_min (1 + N*/N) and the
      fitted critical batch B* = N*/16 as a dashed vertical;
  (b) steps to the target against the rollouts per prompt K at fixed B = 128, with its fit and K* = N*/128.
Both panels share the y-axis range and carry a secondary axis of sequences per step along the top, so the two sweeps can be read
on a common budget axis as well. Data and admission rules are those of plot_steps_vs_seqs_paper.py (KL 1e-3; fastest LR per cell;
pre-fix runs admitted at K = 16 and for K <= 16 at B = 128); the fits are read from its fit table, not refitted here.

Usage: python plot_fig1_prompts_vs_rollouts.py csv/steps_to_50_seqs.csv png/fig1_prompts_vs_rollouts.png
         [--fit-table csv/steps_vs_seqs_b1024_fit_fit_table.csv] [--no-top] [--rollout-b 128|1024] [--one: also write <stem>_one, both sweeps on one sequences axis] [--lr-panel: with --one, also <stem>_with_lr, (a) steps (b) fastest LR, from --lr-table]
Writes the PNG, paper/<stem>.pdf, and the sub-figure files <stem>_a / <stem>_b (0.5 of the line width each, uncropped, equal
height) for a LaTeX subfigure layout; csv/<stem>_table.csv lists the points.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, GREEN, PURPLE = "#1f77b4", "#2ca02c", "#9467bd"   # Figure 1 colours: K = 16 prompt sweep, B = 128 rollout sweep, B = 1024 rollout sweep
TICK, LABEL, LEGEND = 28, 31, 34
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}
FIXED_ONLY_N = {32, 64, 128}
K0 = 16


def opt(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
csv_in, png_out = args[0], args[1]
FIT_TABLE = opt("--fit-table", "csv/steps_vs_seqs_b1024_fit_fit_table.csv", str)
B0 = opt("--rollout-b", 128, int)
LR_TABLE = opt("--lr-table", "csv/steps_to_50_split_table.csv", str)   # fastest LR per cell, written by plot_steps_to_50_combined.py --split   # the rollout sweep's fixed B: 128 (pre-fix K <= 16 admitted) or 1024 (fixed code only)
TOP = "--no-top" not in sys.argv

df = pd.read_csv(csv_in)
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df = df[np.isclose(df["kl"].astype(float), 1e-3)].dropna(subset=["seqs", "lr"]).copy()
df = df[~df["downsample"].astype(bool)]
df = df[(df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[COL].notna()]


def fastest_per(src, key):
    rows = []
    for k, s in src.groupby(key):
        s = s.sort_values([COL, "last_val_step"], na_position="last", ascending=[True, False]); p = s.iloc[0]
        rows.append(dict(x=int(k), seqs=int(p["seqs"]), lr=p["lr"], steps=p[COL], run=p["name"]))
    out = pd.DataFrame(rows).sort_values("x")
    return out.dropna(subset=["steps"])


prompts = fastest_per(df[df["n"] == K0], "bsz")
if B0 == 128:
    rollouts = fastest_per(df[(df["bsz"] == B0) & (df["n"].isin(VALID_N_PREFIX) | (df["fixed"].astype(bool) & df["n"].isin(FIXED_ONLY_N)))], "n")
else:
    rollouts = fastest_per(df[(df["bsz"] == B0) & df["fixed"].astype(bool)], "n")
fits = pd.read_csv(FIT_TABLE).set_index("series")
fp, fr = fits.loc[f"more prompts, K = {K0}"], fits.loc[f"more rollouts, B = {B0}"]
pd.concat([prompts.assign(sweep="prompts"), rollouts.assign(sweep="rollouts")]).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

ylo = min(prompts["steps"].min(), rollouts["steps"].min()) / 1.6
yhi = max(prompts["steps"].max(), rollouts["steps"].max()) * 1.6
fmt = lambda v: f"{int(v) // 1024}k" if v >= 1024 else str(int(v))


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(ylo, yhi)


def panel(ax, pts, fit, per, color, marker, ms, xlabel, star_label, star_y=0.93):
    """one sweep: points, the Figure 1 fit in sequences re-expressed on this panel's x, the critical point as a dashed vertical"""
    smin, nstar = float(fit["s_min"]), float(fit["n_star_seqs"]); star = nstar / per
    xs = np.geomspace(pts["x"].min() / 1.5, pts["x"].max() * 1.5, 200)
    ax.plot(xs, smin * (1 + nstar / (per * xs)), ls="-", lw=4.8, color=color, zorder=3)
    ax.scatter(pts["x"], pts["steps"], s=ms ** 2 * 1.5, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ax.axvline(star, ls=(0, (3, 3)), lw=2.6, color=color, alpha=0.55, zorder=2.3)
    star_txt = f"{star / 1000:.1f}k" if star >= 1000 else f"{star:.0f}"
    ax.annotate(f"{star_label} $\\approx$ {star_txt}", (star, 0.96), xycoords=("data", "axes fraction"),
                xytext=(10, 0), textcoords="offset points", ha="left", va="top", fontsize=LEGEND + 2, color=color, zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=1.0))
    style(ax)
    xs_t = sorted(pts["x"]); major = [x for x in xs_t if int(round(np.log2(x))) % 2 == 0] if len(xs_t) > 8 else xs_t
    ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major])
    minor = [x for x in xs_t if x not in major]
    ax.set_xticks(minor, minor=True); ax.set_xticklabels([""] * len(minor), minor=True); ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(min(xs_t) / 1.9, max(xs_t) * 1.9)
    ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
    if TOP:     # sequences per step along the top, the common budget axis of the two panels
        top = ax.secondary_xaxis("top", functions=(lambda x: x * per, lambda n: n / per))
        tt = [x * per for x in major]
        top.set_xticks(tt); top.set_xticklabels([fmt(t) for t in tt]); top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.set_xlabel("sequences per step", color=INK, fontsize=LABEL); top.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
        for sp in top.spines.values():
            sp.set_visible(False)
    return smin, nstar, star


def perfect(ax, pts, per, smin, nstar):
    xs = np.geomspace(pts["x"].min() / 1.5, pts["x"].max() * 1.5, 100)
    ax.plot(xs, smin * nstar / (per * xs), ls=":", lw=3.2, color=INK2, alpha=0.9, zorder=2.5)


def draw_a(ax):
    smin, nstar, star = panel(ax, prompts, fp, K0, BLUE, "o", 15.5, f"prompts B  (K = {K0})", "$B^*$")
    perfect(ax, prompts, K0, smin, nstar)
    ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)
    if not SHARED_LEGEND:
        hnd = [Line2D([], [], marker="o", ls="", ms=15.5, color=BLUE, mec=SURFACE, label="fastest LR at each B"),
               Line2D([], [], ls="-", lw=4.8, color=BLUE, label="fit  $S = S_\\min\\,(1 + N^*/N)$")]
        ax.legend(handles=hnd, loc="lower left", frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3)


def draw_b(ax):
    col, mk, ms = (GREEN, "p", 17) if B0 == 128 else (PURPLE, "v", 17)
    smin, nstar, star = panel(ax, rollouts, fr, B0, col, mk, ms, f"rollouts K  (B = {B0})", "$K^*$")
    perfect(ax, rollouts, B0, smin, nstar)
    ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)
    if not SHARED_LEGEND:
        hnd = [Line2D([], [], marker=mk, ls="", ms=ms, color=col, mec=SURFACE, label="fastest LR at each K"),
               Line2D([], [], ls="-", lw=4.8, color=col, label="fit  $S = S_\\min\\,(1 + N^*/N)$")]
        ax.legend(handles=hnd, loc="upper right" if B0 != 128 else "lower left", frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3)


def draw_one(ax):
    """both sweeps on one axis of sequences per step (the only x they share): the prompt sweep in B x 16, the rollout sweep in 1024 x K"""
    style(ax)
    rc, rmk = (GREEN, "p") if B0 == 128 else (PURPLE, "v")
    for pts, fit, per, color, mk, ms, lab, star_lab, dy in ((prompts, fp, K0, BLUE, "o", 15.5, f"scaling prompts  (K = {K0})", "$B^*$", 0.96),
                                                            (rollouts, fr, B0, rc, rmk, 17, f"scaling rollouts  (B = {B0})", "$K^*$", 0.96)):
        smin, nstar = float(fit["s_min"]), float(fit["n_star_seqs"]); xs = np.geomspace(pts["seqs"].min() / 1.5, pts["seqs"].max() * 1.5, 200)
        ax.plot(xs, smin * (1 + nstar / xs), ls="-", lw=4.8, color=color, zorder=3)
        ax.scatter(pts["seqs"], pts["steps"], s=ms ** 2 * 1.5, marker=mk, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=6, label=lab)
        ax.axvline(nstar, ls=(0, (3, 3)), lw=2.6, color=color, alpha=0.55, zorder=2.3)
        star = nstar / per; star_txt = f"{star / 1000:.1f}k" if star >= 1000 else f"{star:.0f}"
        # each label hangs off the top of its own line, on the side away from the other line: K* to the left, B* to the right
        ax.annotate(f"{star_lab} $\\approx$ {star_txt}", (nstar, 0.80 if per == K0 else 0.90), xycoords=("data", "axes fraction"), xytext=(24 if per == K0 else -24, 0), textcoords="offset points",
                    ha="center", va="top", fontsize=LEGEND + 2, color=color, zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=1.0))
    # perfect scaling: steps falling as 1/N, the prompt fit's small-batch asymptote S_min N*/N
    smin_p, nstar_p = float(fp["s_min"]), float(fp["n_star_seqs"])
    xs = np.geomspace(prompts["seqs"].min() / 1.5, prompts["seqs"].max() * 1.5, 200)
    ax.plot(xs, smin_p * nstar_p / xs, ls=":", lw=3.2, color=INK2, alpha=0.9, zorder=2.5)
    # the doubling corollary from (B = 1024, K = 16): doubling prompts vs doubling rollouts, both land on 32k sequences per step
    P = prompts.set_index("x"); R = rollouts.set_index("x")
    s0, sp2, sr2 = float(P.loc[1024, "steps"]), float(P.loc[2048, "steps"]), float(R.loc[32, "steps"])
    x0, x1 = 1024 * K0, 2048 * K0
    # the two moves as thick black arrows bowed apart (prompts below the curves, rollouts above) so they are not lost on the lines
    # (the arrows live in the zoom inset; the main panel only marks the zoomed region)
    # zoom inset, lower left: the three cells of the corollary with their step counts
    ins = ax.inset_axes([0.05, 0.06, 0.40, 0.32]); ins.set_facecolor(SURFACE)
    for sp_ in ins.spines.values():
        sp_.set_color(INK2); sp_.set_linewidth(1.4)
    xa, xb = x0 / 1.45, x1 * 4.2
    xs_i = np.geomspace(xa, xb, 60)
    # the two fit curves, very faint, so the cells read against them without the lines competing with the arrows
    ins.plot(xs_i, float(fp["s_min"]) * (1 + float(fp["n_star_seqs"]) / xs_i), ls="-", lw=3.6, color=BLUE, alpha=0.18, zorder=2)
    ins.plot(xs_i, float(fr["s_min"]) * (1 + float(fr["n_star_seqs"]) / xs_i), ls="-", lw=3.6, color=rc, alpha=0.18, zorder=2)
    # the start cell (B = 1024, K = 16) belongs to both sweeps: draw both markers, nudged apart so each shows
    ins.scatter([x0 * 2.0 ** -0.06], [s0], s=15.5 ** 2 * 1.5, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ins.scatter([x0 * 2.0 ** 0.06], [s0], s=17 ** 2 * 1.5, marker=rmk, color=rc, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ins.scatter([x1], [sp2], s=15.5 ** 2 * 1.5, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ins.scatter([x1], [sr2], s=17 ** 2 * 1.5, marker=rmk, color=rc, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    for y1, rad in ((sp2, 0.22), (sr2, -0.22)):
        ins.annotate("", (x1, y1), (x0, s0), zorder=8, arrowprops=dict(arrowstyle="-|>", color=INK, lw=3.4, shrinkA=16, shrinkB=16, mutation_scale=34, connectionstyle=f"arc3,rad={rad}"))
    ins.set_xscale("log", base=2); ins.set_yscale("log"); ins.set_xlim(xa, xb); ins.set_ylim(sp2 / 1.55, s0 * 1.45)
    ins.set_xticks([]); ins.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ins.set_yticks([sp2, sr2]); ins.set_yticklabels([f"{sp2:.0f}", f"{sr2:.0f}"], fontsize=TICK - 6); ins.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # 14 and 20 (21 would overprint 20)
    ins.tick_params(which="both", colors=INK, length=0); ins.grid(False)
    ins.annotate("2$\\times$ rollouts", (x1, sr2), xytext=(16, 0), textcoords="offset points", ha="left", va="center", fontsize=TICK + 1, color=rc, zorder=10)
    ins.annotate("2$\\times$ prompts", (x1, sp2), xytext=(16, 0), textcoords="offset points", ha="left", va="center", fontsize=TICK + 1, color=BLUE, zorder=10)
    # the title as a text box on the main axes, drawn above the curves with an opaque background so nothing shows through it
    ins.set_title("")
    ax.annotate("doubling the sequences", (0.05, 0.06 + 0.32), xycoords="axes fraction", xytext=(0, 14), textcoords="offset points",
                ha="left", va="bottom", fontsize=TICK + 1, color=INK, zorder=12, bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=1.0))
    # the marked window on the main panel hugs the three cells (the inset's own axes are wider, to leave room for its labels)
    wx0, wx1, wy0, wy1 = x0 / 1.25, x1 * 1.25, sp2 / 1.25, s0 * 1.6
    ax.indicate_inset([wx0, wy0, wx1 - wx0, wy1 - wy0], inset_ax=ins, edgecolor=INK2, lw=1.4, alpha=0.8)
    allx = sorted(set(prompts["seqs"]) | set(rollouts["seqs"])); major = [x for x in allx if int(round(np.log2(x))) % 2 == 0]
    ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major]); ax.set_xticks([x for x in allx if x not in major], minor=True)
    ax.set_xticklabels([""] * len([x for x in allx if x not in major]), minor=True); ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(min(allx) / 1.9, max(allx) * 1.9)
    ax.set_xlabel("sequences per step  (prompts $\\times$ rollouts)", color=INK, fontsize=LABEL); ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)
    hnd, lab = ax.get_legend_handles_labels()
    hnd.append(Line2D([], [], ls="-", lw=4.8, color=INK2)); lab.append("fit  $S = S_\\min\\,(1 + N^*/N)$")
    hnd.append(Line2D([], [], ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8)); lab.append("critical batch  $N^*$  ($B^*$ or $K^*$)")
    hnd.append(Line2D([], [], ls=":", lw=3.2, color=INK2)); lab.append("perfect scaling (steps $\\propto 1/N$)")
    if not SHARED_LEGEND:
        ax.legend(hnd, lab, loc="upper right", frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3, markerscale=1.0)
    ax._fig1_legend = (hnd, lab)

def draw_lr(ax):
    """side panel: the fastest LR at each point of both sweeps on the same sequences axis; prompts follow eta ~ B^0.5 up to B*,
    rollouts stay flat. Fastest LRs come from the LR-grid table (csv/steps_to_50_split_table.csv)."""
    style(ax); ax.set_ylim(auto=True)
    lrt = pd.read_csv(LR_TABLE)
    def best(panel, key, per):
        g = lrt[lrt["panel"] == panel].dropna(subset=["steps"]); b = g.loc[g.groupby(key)["steps"].idxmin()].sort_values(key)
        return (b[key].to_numpy(float) * per), b["lr"].to_numpy(float), b[key].to_numpy(float)
    rc, rmk = (GREEN, "p") if B0 == 128 else (PURPLE, "v")
    xp, lp, Bp = best("batch_sweep", "bsz", K0)
    xr, lr_, Kr = best("rollout_sweep", "n", B0)
    nstar_p = float(fp["n_star_seqs"]); nstar_r = float(fr["n_star_seqs"])
    # prompt sweep: power-law fit of eta* against B up to the plateau, with the LR figure's rule (plot_steps_to_50_combined.py):
    # walk back from the largest B while the fastest LR stays equal to the top value; those columns are the plateau and are left out
    k = len(lp)
    while k >= 2 and np.isclose(lp[k - 2], lp[-1]):
        k -= 1
    m = np.zeros(len(lp), bool); m[:k] = True
    lx, ly = np.log(Bp[m]), np.log(lp[m]); slope, icpt = np.polyfit(lx, ly, 1)
    xs = np.geomspace(xp.min() / 1.5, xp[m].max(), 100)
    ax.plot(xs, np.exp(icpt) * (xs / K0) ** slope, ls="-", lw=4.8, color=BLUE, zorder=3)
    ax.plot([xp[m].max(), xp.max() * 1.5], [lp[-1]] * 2, ls=":", lw=4.0, color=BLUE, zorder=3)   # the plateau
    # nudge apart only where the two sweeps have a point in the same column AND the two LRs are close enough to overlap
    lrr = dict(zip(xr, lr_)); close_p = np.array([x in lrr and abs(np.log(lrr[x] / l)) < 0.35 for x, l in zip(xp, lp)])
    lrp = dict(zip(xp, lp)); close_r = np.array([x in lrp and abs(np.log(lrp[x] / l)) < 0.35 for x, l in zip(xr, lr_)])
    ax.scatter(xp * 2.0 ** (-0.18 * close_p), lp, s=15.5 ** 2 * 1.5, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=7, label=f"scaling prompts  (K = {K0})")   # nudged left
    # rollout sweep: its own power-law fit of eta* against K (shallow), with a flat reference at the geometric mean, dotted
    lxr, lyr = np.log(Kr), np.log(lr_); slope_r, icpt_r = np.polyfit(lxr, lyr, 1)
    xs_r = np.geomspace(xr.min() / 1.5, xr.max() * 1.5, 100)
    ax.plot(xs_r, np.exp(icpt_r) * (xs_r / B0) ** slope_r, ls="-", lw=4.8, color=rc, zorder=3)
    flat = np.exp(np.log(lr_).mean())
    ax.plot([xr.min() / 1.5, xr.max() * 1.5], [flat] * 2, ls=":", lw=3.0, color=rc, alpha=0.7, zorder=2.6)
    ax.scatter(xr * 2.0 ** (0.18 * close_r), lr_, s=17 ** 2 * 1.5, marker=rmk, color=rc, edgecolors=SURFACE, linewidths=1.5, zorder=6, label=f"scaling rollouts  (B = {B0})")   # nudged right
    for n_, col in ((nstar_p, BLUE), (nstar_r, rc)):
        ax.axvline(n_, ls=(0, (3, 3)), lw=2.6, color=col, alpha=0.55, zorder=2.3)
    # the two exponent labels at the top, each beside its own critical-batch line and facing away from the other, as in panel (a)
    xb_ = 32 * K0; yb_ = np.exp(icpt) * 32 ** slope         # the blue label above its fit at B = 32, in the open space over the line
    ax.annotate(f"$\\eta^* \\propto B^{{{slope:.2f}}}$", (xb_, yb_), xycoords="data", xytext=(-10, 44), textcoords="offset points",
                ha="center", va="bottom", fontsize=LEGEND + 2, color=BLUE, zorder=10)
    xk_ = xr.min() / 1.5; yk_ = np.exp(icpt_r) * (xk_ / B0) ** slope_r   # the purple label just left of where its fit starts
    ax.annotate(f"$\\eta^* \\propto K^{{{slope_r:.2f}}}$", (xk_, yk_), xycoords="data", xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=LEGEND + 2, color=rc, zorder=10,
                )
    allx = sorted(set(xp) | set(xr)); major = [x for x in allx if int(round(np.log2(x))) % 2 == 0]
    ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major]); ax.set_xticks([x for x in allx if x not in major], minor=True)
    ax.set_xticklabels([""] * len([x for x in allx if x not in major]), minor=True); ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(min(allx) / 1.9, max(allx) * 1.9); ax.set_ylim(min(lp.min(), lr_.min()) / 1.6, max(max(lp.max(), lr_.max()) * 1.8, 1.15e-4))   # keep the 1e-4 tick
    ax.set_xlabel("sequences per step  (prompts $\\times$ rollouts)", color=INK, fontsize=LABEL); ax.set_ylabel("fastest learning rate", color=INK, fontsize=LABEL)
    if not SHARED_LEGEND:
        ax.legend(loc="lower right", frameon=False, fontsize=LEGEND - 6, labelcolor=INK, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.3)


SHARED_LEGEND = False
if "--one" in sys.argv:       # single-panel variant: both sweeps on one sequences-per-step axis; with --lr-panel a second panel of the fastest LRs
    pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
    f1, a1 = plt.subplots(figsize=(15.0, 9.4), facecolor=SURFACE, layout="constrained"); draw_one(a1)
    one = png_out.replace(".png", "_one.png"); f1.savefig(one, dpi=170, facecolor=SURFACE)
    f1.savefig(os.path.join(pdf_dir, os.path.basename(one).replace(".png", ".pdf")), facecolor=SURFACE); plt.close(f1); print("wrote", one)
    if "--lr-panel" in sys.argv:    # (a) steps, (b) fastest LR: a full line width, plus _a / _b sub-figure files at 0.55 / 0.44
        SHARED_LEGEND = True
        f2, (c1, c2) = plt.subplots(1, 2, figsize=(26.0, 10.4), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.3, 1]))
        f2.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.27, wspace=0.24); draw_one(c1); draw_lr(c2)
        hnd, lab = c1._fig1_legend
        f2.legend(hnd, lab, loc="lower center", ncol=3, frameon=False, fontsize=LEGEND - 2, labelcolor=INK, handletextpad=0.6, columnspacing=2.2, bbox_to_anchor=(0.5, 0.0))
        two = png_out.replace(".png", "_with_lr.png"); f2.savefig(two, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        f2.savefig(os.path.join(pdf_dir, os.path.basename(two).replace(".png", ".pdf")), facecolor=SURFACE, bbox_inches="tight"); plt.close(f2)
        for tag, fn, wf in (("a", draw_one, 0.55), ("b", draw_lr, 0.44)):
            f_, a_ = plt.subplots(figsize=(wf * 26.0, 9.4), facecolor=SURFACE, layout="constrained"); fn(a_)
            f_.savefig(two.replace(".png", f"_{tag}.png"), dpi=170, facecolor=SURFACE)
            f_.savefig(os.path.join(pdf_dir, os.path.basename(two).replace(".png", f"_{tag}.pdf")), facecolor=SURFACE); plt.close(f_)
        fL = plt.figure(figsize=(26.0, 2.4), facecolor=SURFACE)     # the shared legend as its own strip, for the LaTeX subfigure layout
        fL.legend(hnd, lab, loc="center", ncol=3, frameon=False, fontsize=LEGEND - 2, labelcolor=INK, handletextpad=0.6, columnspacing=2.2)
        fL.savefig(two.replace(".png", "_legend.png"), dpi=170, facecolor=SURFACE)
        fL.savefig(os.path.join(pdf_dir, os.path.basename(two).replace(".png", "_legend.pdf")), facecolor=SURFACE); plt.close(fL)
        print("wrote", two, "and its _a / _b files")

H = 8.6 if TOP else 7.4
SHARED_LEGEND = True
rc_, rmk_ = (GREEN, "p") if B0 == 128 else (PURPLE, "v")
two_hnd = [Line2D([], [], marker="o", ls="", ms=15.5, color=BLUE, mec=SURFACE), Line2D([], [], marker=rmk_, ls="", ms=17, color=rc_, mec=SURFACE),
           Line2D([], [], ls="-", lw=4.8, color=INK2), Line2D([], [], ls=(0, (3, 3)), lw=2.6, color=INK2, alpha=0.8), Line2D([], [], ls=":", lw=3.2, color=INK2)]
two_lab = [f"scaling prompts  (K = {K0})", f"scaling rollouts  (B = {B0})", "fit  $S = S_\\min\\,(1 + N^*/N)$", "critical batch  $N^*$  ($B^*$ or $K^*$)", "perfect scaling (steps $\\propto 1/N$)"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(26.0, H + 2.6), facecolor=SURFACE)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86 if TOP else 0.96, bottom=0.30, wspace=0.26)
draw_a(axa); draw_b(axb)
fig.legend(two_hnd, two_lab, loc="lower center", ncol=3, frameon=False, fontsize=LEGEND - 2, labelcolor=INK, handletextpad=0.6, columnspacing=2.2, bbox_to_anchor=(0.5, 0.0))
fig.savefig(png_out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
fL2 = plt.figure(figsize=(26.0, 2.4), facecolor=SURFACE)
fL2.legend(two_hnd, two_lab, loc="center", ncol=3, frameon=False, fontsize=LEGEND - 2, labelcolor=INK, handletextpad=0.6, columnspacing=2.2)
fL2.savefig(png_out.replace(".png", "_legend.png"), dpi=170, facecolor=SURFACE)
fL2.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper", os.path.basename(png_out).replace(".png", "_legend.pdf")), facecolor=SURFACE); plt.close(fL2)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), facecolor=SURFACE, bbox_inches="tight")
stem = png_out.replace(".png", "")
for tag, fn in (("a", draw_a), ("b", draw_b)):     # sub-figures: 0.5 of the line width each, uncropped, equal height -> top-align in LaTeX
    f_, a_ = plt.subplots(figsize=(0.5 * 26.0, H), facecolor=SURFACE, layout="constrained"); fn(a_)
    f_.savefig(f"{stem}_{tag}.png", dpi=170, facecolor=SURFACE)
    f_.savefig(os.path.join(pdf_dir, os.path.basename(stem) + f"_{tag}.pdf"), facecolor=SURFACE)
    plt.close(f_)
print("wrote", png_out, "and the _a / _b sub-figure files (PNG + paper PDFs)")
print(f"prompts (K = {K0}): S_min {float(fp['s_min']):.2f}, N* {float(fp['n_star_seqs']):,.0f} seqs = B* {float(fp['n_star_seqs']) / K0:,.0f}; {len(prompts)} points")
print(f"rollouts (B = {B0}): S_min {float(fr['s_min']):.2f}, N* {float(fr['n_star_seqs']):,.0f} seqs = K* {float(fr['n_star_seqs']) / B0:,.0f}; {len(rollouts)} points")
