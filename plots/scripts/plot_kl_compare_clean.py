"""Paper figure: the KL coefficient's effect on the K = 16 prompt sweep, in the style of rollouts_vs_downsample_clean.

  left   steps to the target against batch size B for KL 1e-3 and KL 1e-2 (the fastest LR at each B per KL); with --fit the
         Figure 1 critical-batch fits of both series, read from the Figure 1 fit table (--fit-table) rather than refitted here.
  right  the fastest LR at each B per KL, joined by a solid line, with every other LR tried as a faint dot: filled = reached
         the target, hollow = did not reach it after training at least as long as the fastest LR at that B, x = too early to
         tell (drawn only in columns whose fastest LR is not yet bracketed). The two KLs are nudged apart within a column.

Both series: plain GRPO, K = 16, fastest run per (B, LR, KL); pre-fix and fixed-code runs pooled, as in Figure 1. With --fit each
series' fitted critical batch B* = N*/16 is marked by a dashed vertical and labelled.
--lr-min / --lr-max trim LR rows that only add headroom (nothing outside 5e-7..1.2e-4 ever crossed). The right panel shows only batch
sizes with a crossing at both KLs (so B = 2 and B = 4, which the KL 1e-2 sweep does not cover, are left out).

Usage: python plot_kl_compare_clean.py csv/steps_to_50_kl.csv png/kl_compare_clean.png [--fit] [--fit-table csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv]
                                       [--lr-min 5e-7] [--lr-max 1.2e-4] [--pct 50]
Writes the PNG, paper/<stem>.pdf and csv/<stem>_table.csv (per B and KL: fastest LR, steps, bracket status).
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MISSING = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e8a"
BLUE, LIGHTBLUE = "#1b9e77", "#c2185b"          # KL 1e-3 green, KL 1e-2 pink: the KL-stability figure's pair, unused by any Figure 1 series
TICK, LABEL, TITLE, LEGEND = 28, 31, 32, 34
SHORT_STEPS = 200
K = 16


def opt(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
csv_in, png_out = args[0], args[1]
FIT = "--fit" in sys.argv
FIT_TABLE = opt("--fit-table", "csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv", str)
LR_MIN, LR_MAX, PCT = opt("--lr-min", 5e-7, float), opt("--lr-max", 1.2e-4, float), opt("--pct", 50, int)
df = pd.read_csv(csv_in)
COL = f"steps_to_{PCT}_interp" if f"steps_to_{PCT}_interp" in df else f"steps_to_{PCT}"
df = df[(df["n"] == K)].dropna(subset=["bsz", "lr"]).copy()
df = df[(df["lr"] >= LR_MIN / 1.001) & (df["lr"] <= LR_MAX * 1.001)]
df["last_val_step"] = df["last_val_step"].fillna(0)
SERIES = [(1e-3, "KL 1e-3", BLUE, "o", 15.5), (1e-2, "KL 1e-2", LIGHTBLUE, "s", 13.8)]   # paper marker size (s = 360 for circles)


def ladder(src):
    """one row per (B, LR): fastest crossing, longest run, verdict against the fastest LR at that B; inconclusive runs are
    dropped from columns whose fastest LR already has a definite worse LR on both sides"""
    rows = []
    for (b, lr), s in src.groupby(["bsz", "lr"]):
        h = s.dropna(subset=[COL])
        rows.append(dict(bsz=int(b), lr=float(lr), steps=h[COL].min() if len(h) else np.nan, longest=s["last_val_step"].max()))
    t = pd.DataFrame(rows); out = []
    for b, g in t.groupby("bsz"):
        g = g.sort_values("lr").reset_index(drop=True); hits = g.dropna(subset=["steps"])
        ref = hits["steps"].min() if len(hits) else np.nan
        verdict = []
        for _, r in g.iterrows():
            if not np.isnan(r["steps"]): verdict.append("best" if r["steps"] == ref else "hit")
            elif r["longest"] >= min(ref if not np.isnan(ref) else SHORT_STEPS, SHORT_STEPS): verdict.append("never")
            else: verdict.append("inconclusive")
        g["verdict"] = verdict
        bi = g.index[g["verdict"] == "best"]
        status = "no crossing"
        if len(bi):
            i = bi[0]; definite = lambda j: 0 <= j < len(g) and g.loc[j, "verdict"] in ("hit", "never")
            status = "complete" if definite(i - 1) and definite(i + 1) else ("open" if (i == 0 or i == len(g) - 1) else "inconclusive")
            if status == "complete":
                g = g[g["verdict"] != "inconclusive"]
        g["bracket"] = status
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


lads, bests = {}, {}
for kl, name, *_ in SERIES:
    lad = ladder(df[np.isclose(df["kl"].astype(float), kl)])
    lads[kl] = lad; bests[kl] = lad[lad["verdict"] == "best"].sort_values("bsz")
shown_b = set(bests[1e-3]["bsz"]) & set(bests[1e-2]["bsz"])          # the right panel shows only batch sizes with a crossing at BOTH KLs
for kl in lads:
    lads[kl] = lads[kl][lads[kl]["bsz"].isin(shown_b)].reset_index(drop=True)
table = pd.concat([lads[kl].assign(kl=kl) for kl, *_ in SERIES])
table.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ------------------------------------------------------------------ figure
fig, (ax, axr) = plt.subplots(1, 2, figsize=(26.0, 8.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.22, wspace=0.22)
for a_ in (ax, axr):
    a_.set_facecolor(SURFACE); a_.grid(True, which="major", color=GRID, lw=0.6); a_.set_axisbelow(True)
    a_.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a_.spines[sp].set_color(GRID)

# ---- left: steps to target vs B
fits = pd.read_csv(FIT_TABLE).set_index("series") if FIT and os.path.exists(FIT_TABLE) else None
FIT_KEY = {1e-3: "more prompts, K = 16", 1e-2: "more prompts, K = 16, KL 1e-2"}
for kl, name, color, marker, ms in SERIES:
    b = bests[kl]
    if FIT and fits is not None and FIT_KEY[kl] in fits.index:
        f = fits.loc[FIT_KEY[kl]]; smin, nstar = float(f["s_min"]), float(f["n_star_seqs"])
        xs = np.geomspace(b["bsz"].min() / 1.4, b["bsz"].max() * 1.4, 200)
        ax.plot(xs, smin * (1 + nstar / (K * xs)), ls="-", lw=4.8, color=color, zorder=3)
        bstar = nstar / K                                     # the fitted critical batch in prompts
        ax.axvline(bstar, ls=(0, (3, 3)), lw=2.6, color=color, alpha=0.55, zorder=2.3)
        row = 0 if abs(kl - 1e-3) < 1e-12 else 1
        ax.annotate(f"$B^* \\approx$ {bstar / 1000:.1f}k", (bstar, 0.60 - 0.13 * row), xycoords=("data", "axes fraction"),
                    xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=LEGEND - 6, color=color, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    else:
        ax.plot(b["bsz"], b["steps"], lw=4.8, color=color, zorder=3)
    ax.scatter(b["bsz"], b["steps"], s=ms ** 2 * 1.5, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=6)
ax.set_xscale("log", base=2); ax.set_yscale("log")
bs = sorted(set(bests[1e-3]["bsz"]) | set(bests[1e-2]["bsz"]))
fmt = lambda x: f"{int(x) // 1024}k" if x >= 1024 else str(int(x))
major = [x for x in bs if int(round(np.log2(x))) % 2 == 0]
ax.set_xticks(major); ax.set_xticklabels([fmt(x) for x in major])
ax.set_xticks([x for x in bs if x not in major], minor=True); ax.set_xticklabels([""] * len([x for x in bs if x not in major]), minor=True)
ax.tick_params(axis="x", which="minor", length=0)
ax.set_xlabel(f"prompts B  (K = {K})", color=INK, fontsize=LABEL)
ax.set_ylabel("steps to target", color=INK, fontsize=LABEL)
# no panel title: what is fixed goes in the x labels
legend = [Line2D([], [], marker=m, ls="", ms=ms_, color=c, mec=SURFACE, label=n) for _, n, c, m, ms_ in SERIES]
if FIT and fits is not None:
    legend.append(Line2D([], [], ls="-", lw=4.8, color=INK2, label="fit  $S = S_\\min\\,(1 + N^*/N)$\n(as in Fig. 1)"))
ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=LEGEND - 4, labelcolor=INK, handletextpad=0.6, borderaxespad=0.3)

# ---- right: the fastest LR per B as a solid line with markers; every other LR tried as a faint dot (filled = reached, hollow = not)
for (kl, name, color, marker, ms), nudge in zip(SERIES, (-0.07, +0.07)):
    lad = lads[kl]; x = lad["bsz"] * 2.0 ** nudge
    other = lad[lad["verdict"] != "best"]
    hit = other[other["verdict"] == "hit"]; nev = other[other["verdict"] == "never"]; inc = other[other["verdict"] == "inconclusive"]
    axr.scatter(x[hit.index], hit["lr"], s=110, marker="o", color=color, alpha=0.35, edgecolors="none", zorder=3)
    axr.scatter(x[nev.index], nev["lr"], s=110, marker="o", facecolors="none", edgecolors=color, alpha=0.35, linewidths=1.8, zorder=3)
    axr.scatter(x[inc.index], inc["lr"], s=130, marker="x", c=MISSING, alpha=0.6, linewidths=2.0, zorder=3)
    best = lad[lad["verdict"] == "best"].sort_values("bsz")
    axr.plot(best["bsz"] * 2.0 ** nudge, best["lr"], ls="-", lw=4.8, color=color, zorder=5)
    axr.scatter(best["bsz"] * 2.0 ** nudge, best["lr"], s=ms ** 2 * 1.5, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=6)
axr.set_xscale("log", base=2); axr.set_yscale("log")
bsr = sorted(set(lads[1e-3]["bsz"]) | set(lads[1e-2]["bsz"]))
majr = [x for x in bsr if int(round(np.log2(x))) % 2 == 0]
axr.set_xticks(majr); axr.set_xticklabels([fmt(x) for x in majr])
axr.set_xticks([x for x in bsr if x not in majr], minor=True); axr.set_xticklabels([""] * len([x for x in bsr if x not in majr]), minor=True)
axr.tick_params(axis="x", which="minor", length=0)
axr.set_xlim(min(bsr) / 1.9, max(bsr) * 1.9)
axr.set_xlabel(f"prompts B  (K = {K})", color=INK, fontsize=LABEL)
axr.set_ylabel("learning rate", color=INK, fontsize=LABEL)
legend_r = [Line2D([], [], ls="-", lw=4.8, marker="o", ms=12, color=INK2, mec=SURFACE, label="fastest LR at each B"),
            Line2D([], [], marker="o", ls="", ms=9, color=INK2, alpha=0.45, mec="none", label="other LR: reached the target"),
            Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=INK2, alpha=0.6, mew=1.8, label="other LR: did not reach it")]
if any((lads[kl]["verdict"] == "inconclusive").any() for kl, *_ in SERIES):
    legend_r.append(Line2D([], [], marker="x", ls="", ms=11, color=MISSING, mew=2.0, label="too early to tell"))
axr.legend(handles=legend_r, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=LEGEND - 4, labelcolor=INK,
           handletextpad=0.5, columnspacing=1.6)
fig.savefig(png_out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), facecolor=SURFACE, bbox_inches="tight")
print("wrote", png_out)
for kl, name, *_ in SERIES:
    b = bests[kl]; lad = lads[kl]
    print(f"\n{name}: fastest LR per B")
    for _, r in b.iterrows():
        print(f"  B={r['bsz']:<6d} lr={r['lr']:<8g} steps={r['steps']:7.1f}  bracket={lad.loc[lad['bsz'] == r['bsz'], 'bracket'].iloc[0]}")
