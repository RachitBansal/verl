"""Figure 1 split into three panels (draft layout, not in the paper yet):
  (i)   big, left:     scaling prompts at K = 64  vs  scaling rollouts at B = 1024
  (ii)  top right:     scaling prompts at K = 16  vs  K = 64
  (iii) bottom right:  scaling rollouts at B = 128  vs  B = 1024

Same selection as plot_steps_vs_seqs_paper.py (KL 1e-3, fastest LR per point, per-step-validated reruns replace the
coarse runs of their (K, B, LR) cell, pre-fix runs only where they are honest), same fit S = S_min (1 + N*/N) and the
same critical-batch marker (25% above perfect scaling, N_crit = N*/4). Colours follow Figure 1; B = 256 is new (brown).

Usage: python plot_fig1_panels.py csv/steps_to_50_seqs.csv png/fig1_panels.png [--share-x]
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as PS                                   # shared paper style: sub-figure captions go under each panel

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
COL = dict(k16="#1f77b4", k64="#d66300", b128="#2ca02c", b256="#8c564b", b1024="#9467bd")
MARK = dict(k16="o", k64="h", b128="p", b256="s", b1024="v")
LAB = dict(k16="scaling prompts, K = 16", k64="scaling prompts, K = 64", b128="scaling rollouts, B = 128",
           b256="scaling rollouts, B = 256", b1024="scaling rollouts, B = 1024")
SHORT_STEPS, KNEE_TOL = 200, 1.25
VALID_N_PREFIX = {1, 2, 4, 8, 16}
TOKENS_PER_SEQ = 4096

csv_in, png_out = sys.argv[1], sys.argv[2]
SHARE_X = "--share-x" in sys.argv
df = pd.read_csv(csv_in)
C = "steps_to_50_interp"
df = df[np.isclose(df["kl"].astype(float), 1e-3)].dropna(subset=["seqs", "lr"]).copy()
df["seqs"] = df["seqs"].astype(int)
df = df[~df["downsample"].astype(bool)]
df = df[(df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[C].notna()]
cell = ["n", "bsz", "lr"]
dkey = set(df.loc[df["name"].str.contains("_dense") & df[C].notna(), cell].astype(str).agg("|".join, axis=1))
df = df[~(df[cell].astype(str).agg("|".join, axis=1).isin(dkey) & ~df["name"].str.contains("_dense"))]
fixed = df["fixed"].astype(bool)


def fastest(src, key):
    rows = []
    for k, s in src.groupby(key):
        p = s.sort_values([C, "last_val_step"], na_position="last", ascending=[True, False]).iloc[0]
        rows.append(dict(key=int(k), N=int(p["seqs"]), S=p[C], lr=p["lr"], run=p["name"]))
    return pd.DataFrame(rows).dropna(subset=["S"]).sort_values("N")


def fit(N, S):
    N = np.asarray(N, float); y = np.log(np.asarray(S, float)); best = None
    for ln in np.linspace(np.log(8), np.log(2 ** 26), 6000):
        g = np.log1p(np.exp(ln) / N); a = (y - g).mean(); r = np.sqrt(((y - g - a) ** 2).mean())
        if best is None or r < best[2]:
            best = (float(np.exp(a)), float(np.exp(ln)), float(r))
    return best


SER = {
    "k16": (fastest(df[df["n"] == 16], "bsz"), 16, "B"),
    "k64": (fastest(df[(df["n"] == 64) & fixed], "bsz"), 64, "B"),
    "b128": (fastest(df[(df["bsz"] == 128) & (df["n"].isin(VALID_N_PREFIX) | fixed)], "n"), 128, "K"),
    "b256": (fastest(df[(df["bsz"] == 256) & (df["n"].isin(VALID_N_PREFIX) | fixed)], "n"), 256, "K"),
    "b1024": (fastest(df[(df["bsz"] == 1024) & fixed], "n"), 1024, "K"),
}
FITS = {}
for k, (d, per, unit) in SER.items():
    smin, ns, rms = fit(d["N"], d["S"])
    FITS[k] = dict(smin=smin, ns=ns, rms=rms, nc=(KNEE_TOL - 1) * ns, per=per, unit=unit, n=len(d))
    print(f"{k:6s} {len(d):2d} pts  S_min={smin:6.2f}  N*={ns:9.0f}  N_crit={(KNEE_TOL - 1) * ns:8.0f} ({unit}_crit={(KNEE_TOL - 1) * ns / per:.3g})  rms={rms:.3f}")
    print("        " + "  ".join(f"{unit}={r.key}:{r.S:.1f}" for r in d.itertuples()))


def fmt_k(v):
    return f"{v / 1000:.2g}k" if v >= 1000 else f"{float(f'{v:.2g}'):.0f}"


# Sized for \includegraphics[width=\linewidth] (about 5.5 in): the canvas is 13.5 in wide, so everything shrinks by ~0.41
# and these sizes land at roughly 8-10 pt in the paper; lines at ~1.5-2 pt.
FS = dict(big=dict(title=25, label=24, tick=21, legend=21, annot=23, line=4.4, dot=2.8, crit=3.2, s=260, ring=17),
          small=dict(title=22, label=21, tick=19, legend=19, annot=20, line=3.8, dot=2.6, crit=2.8, s=170, ring=14))
plt.rcParams.update({"font.size": 20})
fig = plt.figure(figsize=(13.5, 8.6), facecolor=SURFACE, layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1])
axI = fig.add_subplot(gs[:, 0]); axII = fig.add_subplot(gs[0, 1]); axIII = fig.add_subplot(gs[1, 1])
PANELS = [(axI, ["k64", "b1024"], "(a) Prompts at $K=64$ vs. rollouts at $B=1024$."),
          (axII, ["k16", "k64"], "(b) Prompts: $K=16$ vs. $K=64$."),
          (axIII, ["b128", "b1024"], "(c) Rollouts: $B=128$ vs. $B=1024$.")]
SHORT = dict(k16="K = 16", k64="K = 64", b128="B = 128", b256="B = 256", b1024="B = 1024")
allN = np.concatenate([SER[k][0]["N"].to_numpy() for k in SER]); allS = np.concatenate([SER[k][0]["S"].to_numpy() for k in SER])
YLIM = (allS.min() / 2.2, allS.max() * 1.8)
OFFSET = {"k16": (-14, -30), "b1024_small": (-14, -30)}     # (ii): K = 16 label below-left of its circle; (iii): B = 1024 label below-right


def draw(ax, keys, title, big):
    z = FS["big" if big else "small"]
    xs_all = np.concatenate([SER[k][0]["N"].to_numpy() for k in keys])
    xlo, xhi = (allN.min(), allN.max()) if SHARE_X else (xs_all.min(), xs_all.max())
    for k in keys:
        d, per, unit = SER[k]; f = FITS[k]; c = COL[k]
        p0 = d.iloc[0]
        xs = np.array([xlo / 1.6, xhi * 1.6])
        ax.plot(xs, p0["S"] * p0["N"] / xs, ls=":", lw=z["dot"], color=c, alpha=0.6, zorder=1)          # perfect scaling from the smallest point
        fx = np.geomspace(d["N"].min() / 1.4, d["N"].max() * 1.4, 200)
        ax.plot(fx, f["smin"] * (1 + f["ns"] / fx), "-", lw=z["line"], color=c, alpha=0.9, zorder=3)
        ax.scatter(d["N"], d["S"], s=z["s"], marker=MARK[k], color=c, edgecolors=SURFACE, linewidths=1.5, zorder=5)
        nc = f["nc"]; sc = f["smin"] * (1 + f["ns"] / nc)
        if not (d["N"].min() <= nc <= d["N"].max()):     # knee outside the measured range: not marked
            print(f"{k}: N_crit {nc:.0f} outside its data {d['N'].min()}..{d['N'].max()}, not marked")
            continue
        ax.axvline(nc, ls=":", lw=z["crit"], color=c, zorder=2)
        ax.plot([nc], [sc], marker="o", ms=z["ring"], mfc=SURFACE, mec=c, mew=3.0, ls="", zorder=9)
        off = OFFSET.get(k + ("" if big else "_small"), OFFSET.get(k, (12, 12)))
        ax.annotate(f"${unit}_\\mathrm{{crit}} \\approx {fmt_k(nc / per)}$", xy=(nc, sc), xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left", fontsize=z["annot"], color=c, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ys_all = np.concatenate([SER[k][0]["S"].to_numpy() for k in keys])
    ax.set_xlim(xlo / 1.6, xhi * 1.6); ax.set_ylim(ys_all.min() / 2.0, ys_all.max() * (3.0 if big else 4.0))   # each panel its own y-range (headroom for the legend)
    ticks = [2 ** i for i in range(4, 20) if xlo / 1.6 <= 2 ** i <= xhi * 1.6]
    step = 1 if len(ticks) <= 6 else 2
    ticks = [t for t in ticks if int(np.log2(t)) % step == int(np.log2(ticks[-1])) % step]   # keep the largest batch labelled
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" if t < 1024 else f"{t // 1024}k" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="both", labelsize=z["tick"])
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    PS.subcaption(ax, title, xlabel="sequences per step  $N = B \\times K$", fontsize=PS.FS["label"])
    ax.legend(handles=[Line2D([], [], marker=MARK[k], ls="-", lw=z["line"], ms=14 if big else 12, color=COL[k], mec=SURFACE,
                              label=LAB[k] if big else SHORT[k]) for k in keys],
              loc="upper right", frameon=False, fontsize=z["legend"], handlelength=1.8, borderaxespad=0.2)


for ax, keys, title in PANELS:
    draw(ax, keys, title, big=ax is axI)
axI.set_ylabel("steps to target", fontsize=FS["big"]["label"])

refs = [Line2D([], [], ls=":", lw=FS["big"]["dot"], color=INK2, label="perfect scaling"),
        Line2D([], [], ls="-", lw=FS["big"]["line"], color=INK2, label="fit S = S_min(1 + N*/N)"),
        Line2D([], [], marker="o", ms=15, mfc=SURFACE, mec=INK2, mew=3, ls=":", lw=FS["big"]["crit"], color=INK2, label="critical batch (25% above perfect)")]
fig.legend(handles=refs, loc="outside lower center", ncol=3, frameon=False, fontsize=FS["big"]["legend"] - 1, columnspacing=1.0, handlelength=1.8, handletextpad=0.4)
fig.savefig(png_out, dpi=170, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), facecolor=SURFACE)
print("wrote", png_out)
