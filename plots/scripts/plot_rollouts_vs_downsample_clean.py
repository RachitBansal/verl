"""Paper figure (clean): more rollouts vs upsampled rollouts at B = 128 prompts (the runs are tagged "downsample" in wandb:
they generate 64 and train on K; the paper calls this upsampling the generation side), steps to 50% against
sequences trained per optimizer step. Single panel, Figure 1 styling; no LR ladder or bracket diagnostics
(those stay in plot_rollouts_vs_downsample_paper.py).

  more rollouts  - plain GRPO, B = 128, K rollouts generated and trained per prompt (fastest LR at each K);
                   K = 32/64 from the fixed-code reruns only, K <= 16 pooled with the pre-fix runs
  upsampling     - 64 rollouts generated per prompt, the K with the largest |advantage| kept for training,
                   B = 128, fixed code only (fastest LR at each K)

x = sequences trained per step = 128 K for both series, so they are compared at equal training cost per
step; the upsampling series always generates 128 x 64 = 8,192 sequences per step. Points where no LR
reaches 50% are left out (listed on stdout and in the table). --fit draws the Figure 1 fit of the rollout sweep,
S = S_min (1 + N*/N) with S_min and N* read from its fit table (--fit-table, default csv/steps_vs_seqs_fit_fit_table.csv;
that fit uses every B = 128 column, including K = 128, which this figure leaves out), and marks N*. The upsampling
series is not fitted: four points are too few.

A right-hand panel shows every LR tried at each K for both series, with the fastest ringed (filled = reached the target,
hollow = did not reach it after training at least as long as the fastest LR, x = too early to tell, drawn only where the fastest LR is not yet bracketed), to make visible that the LR
choice barely moves between the two.

Usage: python plot_rollouts_vs_downsample_clean.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_clean.png [--fit]
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

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
GREEN, PURPLE = "#2ca02c", "#c2378f"     # tab:green = the rollout sweep's colour in Figure 1; magenta for upsampling (unused elsewhere; purple is the B = 1024 sweep). Green/magenta stay apart under protan, deutan and tritan simulation (CIE76 92/56/102)
B = 128
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}        # K where a pre-fix run is an honest measurement
FIXED_ONLY_N = {32, 64}                  # K where only fixed-code reruns count; K = 128 has no upsampling counterpart

csv_in, png_out = sys.argv[1], sys.argv[2]
FIT = "--fit" in sys.argv[3:]
# --fit draws the Figure 1 fit of the rollout sweep (every B = 128 column, including the K = 128 point this figure leaves out)
# taken from its fit table rather than refitted on the points drawn here; the upsampling series has too few points to fit
FIT_TABLE = sys.argv[sys.argv.index("--fit-table") + 1] if "--fit-table" in sys.argv else "csv/steps_vs_seqs_fit_fit_table.csv"
df = pd.read_csv(csv_in)
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df = df[np.isclose(df["kl"].astype(float), 1e-3) & (df["bsz"] == B)].dropna(subset=["lr"]).copy()
df["last_val_step"] = df["last_val_step"].fillna(0)
df["downsample"] = df["downsample"].astype(bool); df["fixed"] = df["fixed"].astype(bool)


def fastest_per_k(src, key):
    """per K: the fastest crossing over every LR; Ks with no crossing are returned separately with their best accuracy."""
    hits, never = [], []
    for k, s in src.groupby(key):
        h = s.dropna(subset=[COL])
        if len(h):
            r = h.sort_values(COL).iloc[0]
            hits.append(dict(k=int(k), seqs=B * int(k), lr=r["lr"], steps=r[COL], run=r["name"]))
        else:
            never.append(dict(k=int(k), seqs=B * int(k), best_val=s["max_val"].max(), longest=s["last_val_step"].max(), attempts=len(s)))
    return pd.DataFrame(hits).sort_values("seqs"), pd.DataFrame(never)


SHORT_STEPS = 200


def ladder(src, key):
    """one row per (K, LR): fastest crossing, longest run, and the verdict against the fastest LR at that K
    (hit / "never" = ran at least min(best crossing, SHORT_STEPS) steps without crossing, shown as "did not reach" / inconclusive = too short to judge)."""
    rows = []
    for (k, lr), s_ in src.groupby([key, "lr"]):
        h = s_.dropna(subset=[COL])
        rows.append(dict(k=int(k), lr=float(lr), steps=h[COL].min() if len(h) else np.nan, longest=s_["last_val_step"].max()))
    t = pd.DataFrame(rows)
    out = []
    for k, g in t.groupby("k"):
        hits = g.dropna(subset=["steps"])
        ref = hits["steps"].min() if len(hits) else np.nan
        for _, r in g.iterrows():
            if not np.isnan(r["steps"]):
                v = "best" if r["steps"] == ref else "hit"
            elif r["longest"] >= min(ref if not np.isnan(ref) else SHORT_STEPS, SHORT_STEPS):
                v = "never"
            else:
                v = "inconclusive"
            out.append(dict(k=int(k), lr=r["lr"], steps=r["steps"], verdict=v))
    out = pd.DataFrame(out)
    # "too early to tell" runs matter only where the fastest LR is not yet bracketed: once the LR just below and just above the
    # fastest are both definite (crossed, or never), the column is settled and its inconclusive runs are dropped
    keep = []
    for k, g in out.groupby("k"):
        g = g.sort_values("lr").reset_index(drop=True)
        b = g.index[g["verdict"] == "best"]
        if len(b):
            i = b[0]
            definite = lambda j: 0 <= j < len(g) and g.loc[j, "verdict"] in ("hit", "never")
            if definite(i - 1) and definite(i + 1):
                g = g[g["verdict"] != "inconclusive"]
        keep.append(g)
    return pd.concat(keep).reset_index(drop=True)


def fit_cbs(seqs, steps):
    """S = S_min (1 + N*/N), least squares in log S (same search as plot_steps_vs_seqs_paper.py)."""
    N = np.asarray(seqs, float); y = np.log(np.asarray(steps, float))
    best = None
    for ln_ns in np.linspace(np.log(8), np.log(2 ** 26), 6000):
        g = np.log1p(np.exp(ln_ns) / N)
        ln_smin = (y - g).mean()
        rms = np.sqrt(((y - g - ln_smin) ** 2).mean())
        if best is None or rms < best[2]:
            best = (float(np.exp(ln_smin)), float(np.exp(ln_ns)), float(rms))
    return best


def fmt_k(v):
    return f"{v / 1000:.0f}k" if v >= 1e5 else (f"{v / 1000:.2g}k" if v >= 1000 else f"{float(f'{v:.2g}'):.0f}")


roll_src = df[(~df["downsample"]) & (df["n"].isin(VALID_N_PREFIX) | (df["fixed"] & df["n"].isin(FIXED_ONLY_N)))]
roll, roll_never = fastest_per_k(roll_src, "n")
ds, ds_never = fastest_per_k(df[df["downsample"] & df["fixed"]], "dsk")
roll_lad, ds_lad = ladder(roll_src, "n"), ladder(df[df["downsample"] & df["fixed"]], "dsk")
LR_MAX = 5e-5                                               # the 1e-4 runs never crossed and every column is bracketed below them
shown_k = set(roll["k"]) | set(ds["k"])                     # K columns on the left panel; K with no crossing in either series are left out
roll_lad, ds_lad = roll_lad[roll_lad["k"].isin(shown_k)].reset_index(drop=True), ds_lad[ds_lad["k"].isin(shown_k)].reset_index(drop=True)
roll_lad, ds_lad = roll_lad[roll_lad["lr"] <= LR_MAX * 1.001].reset_index(drop=True), ds_lad[ds_lad["lr"] <= LR_MAX * 1.001].reset_index(drop=True)

# table: the plotted points plus the Ks without a crossing
rows = []
for series, hits, never in [("more_rollouts", roll, roll_never), ("downsample", ds, ds_never)]:
    for _, r in hits.iterrows():
        rows.append(dict(series=series, k=r["k"], seqs=r["seqs"], lr=r["lr"], steps=r["steps"], run=r["run"]))
    for _, r in never.iterrows():
        rows.append(dict(series=series, k=r["k"], seqs=r["seqs"], lr=np.nan, steps=np.nan, run=f"no crossing (best {r['best_val']:.1%} in {int(r['longest'])} steps, {int(r['attempts'])} runs)"))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ------------------------------------------------------------------ figure (styled like plot_steps_to_50_combined.py)
TICK, LABEL, TITLE, NOTE, LEGEND = 28, 31, 32, 29, 27       # type sizes, as in the steps_to_50 figures
# two panels side by side: steps to target on the left, every LR tried at each K on the right (ring = fastest), to show the
# LR choice barely moves between the two series
MISSING = "#8f8e8a"
fits = []


def style_axes(a_):
    a_.set_facecolor(SURFACE); a_.grid(True, which="major", color=GRID, lw=0.6); a_.set_axisbelow(True)
    a_.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a_.spines[sp].set_color(GRID)


def draw_left(ax):
    """steps to target vs K for both series, the Figure 1 fit of the rollout sweep, sequences per step on top, legend inside"""
    style_axes(ax)
    series = [("more rollouts: generate K, train K", roll, GREEN, "p", 19),
              ("upsampling: generate 64, train the K largest-|advantage|", ds, PURPLE, "D", 16)]
    for label, pts, color, marker, ms in series:
        if FIT and pts is roll:
            ft = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(FIT_TABLE)) if not os.path.isabs(FIT_TABLE) and not os.path.exists(FIT_TABLE) else FIT_TABLE)
            f1 = ft[ft["series"] == f"more rollouts, B = {B}"].iloc[0]
            smin, nstar = float(f1["s_min"]), float(f1["n_star_seqs"])
            xs = np.geomspace(pts["seqs"].min() / 1.4, pts["seqs"].max() * 1.4, 200)
            ax.plot(xs, smin * (1 + nstar / xs), ls="-", lw=4.8, color=color, zorder=3)
            fits.append(dict(series=label, S_min=smin, N_crit=nstar, K_crit=nstar / B, rms_log=float(f1["rms_log"]), points=int(f1["n_points"]), source=FIT_TABLE))
        else:                                   # no fit for this series: join its points
            ax.plot(pts["seqs"], pts["steps"], lw=4.8, color=color, zorder=3)
        ax.scatter(pts["seqs"], pts["steps"], s=ms ** 2 * 1.5, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    if FIT:
        pd.DataFrame(fits).to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_fit_table.csv")), index=False)

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ticks = sorted(set(roll["seqs"]) | set(ds["seqs"]))
    ax.set_xticks(ticks); ax.set_xticklabels([str(t // B) for t in ticks])            # bottom: K (the data stay in sequences = K x B)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(f"trained rollouts K  (B = {B})", color=INK, fontsize=LABEL)
    ax.set_xlim(min(ticks) / 1.9, max(ticks) * 1.9)     # same K range as the right panel (x is in sequences = K x B)
    ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)
    # (the sequences-per-step axis along the top was dropped 2026-10-22: K x B = 128 is in the x label)
    legend = [Line2D([], [], marker="p", ls="", ms=19, color=GREEN, mec=SURFACE, label="generate K, train K"),
              Line2D([], [], marker="D", ls="", ms=16, color=PURPLE, mec=SURFACE, label="generate 64, train K")]
    if FIT:
        legend += [Line2D([], [], ls="-", lw=4.8, color=GREEN, label="fit (Figure 2)")]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=LEGEND, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3)   # the upper right is empty: steps fall with N


def draw_right(axr):
    """every LR tried at each K for both series; ring = fastest; legend under the panel"""
    style_axes(axr)
    # ---- right panel: every LR tried at each K; filled = reached the target, hollow = did not reach it after training at least as long as the fastest LR, x = too early to tell; ring = fastest
    for lad, color, marker, ms, nudge in [(roll_lad, GREEN, "p", 19, -0.15), (ds_lad, PURPLE, "D", 16, +0.15)]:
        x = lad["k"] * 2.0 ** nudge                       # the two series side by side within each K column
        hit = lad[lad["verdict"].isin(["hit", "best"])]; nev = lad[lad["verdict"] == "never"]; inc = lad[lad["verdict"] == "inconclusive"]
        axr.scatter(x[hit.index], hit["lr"], s=ms ** 2 * 1.5, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=5)
        axr.scatter(x[nev.index], nev["lr"], s=ms ** 2 * 1.5, marker=marker, facecolors="none", edgecolors=color, linewidths=2.6, zorder=5)
        axr.scatter(x[inc.index], inc["lr"], s=220, marker="x", c=MISSING, linewidths=2.8, zorder=5)
        best = lad[lad["verdict"] == "best"]
        axr.scatter(x[best.index], best["lr"], s=1050, marker="o", facecolors="none", edgecolors=INK, linewidths=3.8, zorder=6)
    axr.set_xscale("log", base=2); axr.set_yscale("log")
    ks = sorted(set(roll_lad["k"]) | set(ds_lad["k"]))
    axr.set_xticks(ks); axr.set_xticklabels([str(k) for k in ks])
    axr.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axr.set_xlim(min(ks) / 1.9, max(ks) * 1.9)
    axr.set_xlabel(f"trained rollouts K  (B = {B})", color=INK, fontsize=LABEL)
    axr.set_ylabel("learning rate", color=INK, fontsize=LABEL)
    legend_r = [Line2D([], [], marker="o", ls="", ms=26.5, mfc="none", mec=INK, mew=3.8, label="fastest LR"),
                Line2D([], [], marker="o", ls="", ms=18.5, mfc="none", mec=MISSING, mew=2.8, label="did not reach target")]
    if (roll_lad["verdict"] == "inconclusive").any() or (ds_lad["verdict"] == "inconclusive").any():
        legend_r.append(Line2D([], [], marker="x", ls="", ms=15, color=MISSING, mew=2.6, label="too early to tell"))
    global UPS_B_LEGEND
    UPS_B_LEGEND = legend_r    # drawn as a separate legend strip under both sub-figures (rollouts_vs_downsample_clean_fit_legend)


# combined figure (a full line width) and the two sub-figure files (a: 0.53, b: 0.46 of the line width; not cropped, equal height, so
# they top-align when LaTeX sets them side by side, as for the LR split figure)
fig, (ax, axr) = plt.subplots(1, 2, figsize=(PAPER_SCALE * (26.0), PAPER_SCALE * (6.6)), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.22, wspace=0.22)
draw_left(ax); draw_right(axr)
if FIT:
    pd.DataFrame(fits).to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_fit_table.csv")), index=False)
fig.savefig(png_out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper")   # plots/paper/
os.makedirs(pdf_dir, exist_ok=True)
pdf_out = os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf"))
fig.savefig(pdf_out, facecolor=SURFACE, bbox_inches="tight")
stem = png_out.replace(".png", "")
for tag, fn, wfrac in (("a", draw_left, 0.53), ("b", draw_right, 0.46)):
    f_, a_ = plt.subplots(figsize=(PAPER_SCALE * (wfrac * 26.0), PAPER_SCALE * (6.6)), facecolor=SURFACE, layout="constrained"); fn(a_)
    f_.savefig(f"{stem}_{tag}.png", dpi=170, facecolor=SURFACE)
    f_.savefig(os.path.join(pdf_dir, os.path.basename(stem) + f"_{tag}.pdf"), facecolor=SURFACE)
    plt.close(f_)
figL = plt.figure(figsize=(PAPER_SCALE * 26.0, PAPER_SCALE * 1.4), facecolor=SURFACE)    # legend strip for panel (b), under both sub-figures
figL.legend(handles=UPS_B_LEGEND, loc="center", ncol=len(UPS_B_LEGEND), frameon=False, fontsize=LEGEND, labelcolor=INK, handletextpad=0.5, columnspacing=1.6)
figL.savefig(f"{stem}_legend.png", dpi=170, facecolor=SURFACE); figL.savefig(os.path.join(pdf_dir, os.path.basename(stem) + "_legend.pdf"), facecolor=SURFACE); plt.close(figL)
print("wrote", png_out, "and", pdf_out, "plus the _a / _b sub-figure files")
for name, hits, never in [("more rollouts", roll, roll_never), ("upsampling", ds, ds_never)]:
    print(f"\n{name}:")
    for _, r in hits.iterrows():
        print(f"  K={r['k']:<3d} seqs={r['seqs']:<5d} lr={r['lr']:<8g} steps={r['steps']:7.1f}  {r['run']}")
    for _, r in never.iterrows():
        print(f"  K={int(r['k']):<3d} seqs={int(r['seqs']):<5d} no run reaches 50% (best {r['best_val']:.1%} after {int(r['longest'])} steps, {int(r['attempts'])} runs)")
if FIT:
    for f in fits:
        print(f"\nfit {f['series']}: S_min={f['S_min']:.1f}  N_crit={f['N_crit']:.0f} (K={f['K_crit']:.1f})  rms log resid={f['rms_log']:.3f} over {f['points']} points")
