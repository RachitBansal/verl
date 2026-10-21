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


plt.rcParams.update({"font.size": 15, "axes.labelsize": 16})
fig = plt.figure(figsize=(15.5, 8.2), facecolor=SURFACE, layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
axI = fig.add_subplot(gs[:, 0]); axII = fig.add_subplot(gs[0, 1]); axIII = fig.add_subplot(gs[1, 1])
PANELS = [(axI, ["k64", "b1024"], "(i)  prompts at K = 64  vs  rollouts at B = 1024"),
          (axII, ["k16", "k64"], "(ii)  prompts: K = 16  vs  K = 64"),
          (axIII, ["b128", "b1024"], "(iii)  rollouts: B = 128  vs  B = 1024")]
allN = np.concatenate([SER[k][0]["N"].to_numpy() for k in SER]); allS = np.concatenate([SER[k][0]["S"].to_numpy() for k in SER])
YLIM = (allS.min() / 2.2, allS.max() * 1.8)


OFFSET = {"k16": (-12, -26), "b1024_small": (12, -26)}         # in (ii) the K = 16 label goes below-left of its circle, clear of the K = 64 one


def draw(ax, keys, title, big):
    ms = 1.0 if big else 0.75
    xs_all = np.concatenate([SER[k][0]["N"].to_numpy() for k in keys])
    xlo, xhi = (allN.min(), allN.max()) if SHARE_X else (xs_all.min(), xs_all.max())
    for k in keys:
        d, per, unit = SER[k]; f = FITS[k]; c = COL[k]
        p0 = d.iloc[0]
        xs = np.array([xlo / 1.6, xhi * 1.6])
        ax.plot(xs, p0["S"] * p0["N"] / xs, ls=":", lw=1.3, color=c, alpha=0.6, zorder=1)          # perfect scaling from the smallest point
        fx = np.geomspace(d["N"].min() / 1.4, d["N"].max() * 1.4, 200)
        ax.plot(fx, f["smin"] * (1 + f["ns"] / fx), "-", lw=2.2 if big else 1.8, color=c, alpha=0.9, zorder=3)
        ax.scatter(d["N"], d["S"], s=(130 if big else 75), marker=MARK[k], color=c, edgecolors=SURFACE, linewidths=1.2, zorder=5)
        nc = f["nc"]; sc = f["smin"] * (1 + f["ns"] / nc)
        if not (d["N"].min() <= nc <= d["N"].max()):     # knee outside the measured range (B = 256: three points from K = 16): not marked
            print(f"{k}: N_crit {nc:.0f} outside its data {d['N'].min()}..{d['N'].max()}, not marked")
            continue
        ax.axvline(nc, ls=":", lw=2.4 if big else 1.8, color=c, zorder=2)
        ax.plot([nc], [sc], marker="o", ms=13 * ms, mfc=SURFACE, mec=c, mew=2.2, ls="", zorder=9)
        ax.annotate(f"${unit}_\\mathrm{{crit}} \\approx {fmt_k(nc / per)}$", xy=(nc, sc), xytext=OFFSET.get(k + ("" if big else "_small"), OFFSET.get(k, (10, 10))), textcoords="offset points",
                    ha="right" if OFFSET.get(k + ("" if big else "_small"), OFFSET.get(k, (10, 10)))[0] < 0 else "left", fontsize=16 if big else 13, color=c, zorder=10, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlim(xlo / 1.6, xhi * 1.6); ax.set_ylim(*YLIM)
    ticks = [2 ** i for i in range(4, 20) if xlo / 1.6 <= 2 ** i <= xhi * 1.6]
    step = 1 if (big or len(ticks) <= 7) else 2
    ax.set_xticks(ticks[::step]); ax.set_xticklabels([f"{t:,}" if t < 1024 else f"{t // 1024}k" for t in ticks[::step]], fontsize=13 if big else 11.5)
    ax.tick_params(axis="y", labelsize=13 if big else 11.5)
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=17 if big else 14, color=INK)
    ax.legend(handles=[Line2D([], [], marker=MARK[k], ls="-", lw=2, ms=9 if big else 7, color=COL[k], mec=SURFACE,
                              label=LAB[k] + (f"  ({FITS[k]['n']} points, K = 16–64)" if k == "b256" else "")) for k in keys],
              loc="lower left" if big else "upper right", frameon=False, fontsize=14 if big else 11.5)


for ax, keys, title in PANELS:
    draw(ax, keys, title, big=ax is axI)
axI.set_ylabel("steps to target AIME 1983–2024"); axI.set_xlabel("sequences per optimizer step  N = B × K")
axIII.set_xlabel("sequences per optimizer step  N = B × K", fontsize=13)
refs = [Line2D([], [], ls=":", lw=1.3, color=INK2, label="perfect scaling from the smallest batch"),
        Line2D([], [], ls="-", lw=2, color=INK2, label="fit  S = S_min (1 + N*/N)"),
        Line2D([], [], marker="o", ms=10, mfc=SURFACE, mec=INK2, mew=2, ls=":", lw=1.6, color=INK2, label="critical batch, 25% above perfect scaling")]
fig.legend(handles=refs, loc="outside lower center", ncol=3, frameon=False, fontsize=13)
fig.savefig(png_out, dpi=170, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), facecolor=SURFACE)
print("wrote", png_out)
