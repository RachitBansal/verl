"""Steps to reach 50% AIME accuracy vs batch size, KL coef 1e-3 vs 1e-2 (n = 16).

Per KL coefficient and batch size, the headline curve is the fastest learning
rate tried. Every (batch, LR) run that reached 50% is also shown as a small
point colored by LR, with marker shape giving the KL coefficient.
Duplicate runs of one config: if a "_v2" run is among them, one run is drawn at
random (seeded); otherwise the fastest run is kept. Runs with fewer than
SHORT_STEPS validated steps are excluded.

Usage: python plot_steps_vs_bsz_kl.py steps_to_50_kl.csv out.png [seed]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue", RAMP)   # light = small LR, dark = large LR
MUTED = "#a8a7a2"
SHORT_STEPS = 200

SERIES = {  # marker, curve style, label placement (1e-3 lower-left, 1e-2 upper-right)
    1e-3: dict(marker="o", ls="-", color=INK, off=(-9, -15), ha="right", label="KL coef 1e-3", s=42),
    1e-2: dict(marker="D", ls="--", color=INK2, off=(9, 8), ha="left", label="KL coef 1e-2", s=34),
}

csv_in, png_out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed)

df = pd.read_csv(csv_in).dropna(subset=["bsz", "lr"])
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"   # interpolated crossing when available
df = df[df["n"] == 16]
# short runs are excluded only while they have not reached 50% (a hit is a hit)
df = df[(df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[COL].notna()]
df["bsz"] = df["bsz"].astype(int)
df["kl"] = df["kl"].round(6)

rows, choices = [], []
for (kl, bsz, lr), s in df.groupby(["kl", "bsz", "lr"]):
    if len(s) > 1 and s["name"].str.contains("_v2").any():
        pick = s.iloc[rng.integers(len(s))]
        choices.append((kl, bsz, lr, len(s), pick["name"], pick[COL]))
    elif len(s) > 1:
        pick = s.sort_values(COL, na_position="last").iloc[0]
    else:
        pick = s.iloc[0]
    rows.append(dict(kl=kl, bsz=bsz, lr=lr, steps=pick[COL], run=pick["name"],
                     id=pick["id"], n_candidates=len(s),
                     fixed="updated_scaling" in str(pick["name"])))
g = pd.DataFrame(rows)
g.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)
hit = g.dropna(subset=["steps"])
lrnorm = LogNorm(vmin=hit["lr"].min(), vmax=hit["lr"].max())

fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

bests = {}
for kl, st in SERIES.items():
    h = hit[np.isclose(hit["kl"], kl)]
    if h.empty:
        continue
    ax.scatter(h["bsz"], h["steps"], s=st["s"], marker=st["marker"], c=h["lr"], cmap=CMAP,
               norm=lrnorm, edgecolors=SURFACE, linewidths=1.0, zorder=3)
    best = h.loc[h.groupby("bsz")["steps"].idxmin()].sort_values("bsz")
    bests[kl] = best
    ax.plot(best["bsz"], best["steps"], ls=st["ls"], lw=2, color=st["color"], zorder=4)
    ax.scatter(best["bsz"], best["steps"], s=120, marker=st["marker"], facecolors=SURFACE,
               edgecolors=st["color"], linewidths=2, zorder=5)
    for _, r in best.iterrows():
        ax.annotate(f"{r['steps']:.0f}", (r["bsz"], r["steps"]), xytext=st["off"],
                    textcoords="offset points", ha=st["ha"], fontsize=8.2,
                    color=st["color"], zorder=6)

# post-fix runs: small ink dot at the centre (drawn last so it shows on the headline markers too)
fx = hit[hit["fixed"]]
ax.scatter(fx["bsz"], fx["steps"], s=10, c=INK, zorder=7)

# perfect scaling reference anchored at the smallest-batch KL 1e-3 point
b3 = bests[1e-3]
b0, s0 = b3["bsz"].iloc[0], b3["steps"].iloc[0]
xs = np.array([b3["bsz"].min() / 1.3, b3["bsz"].max() * 1.3])
ax.plot(xs, s0 * b0 / xs, ls=":", lw=1.3, color=MUTED, zorder=2)
ax.annotate("perfect scaling (steps ∝ 1/batch)", (xs[1], s0 * b0 / xs[1]), xytext=(-4, -6), va="top",
            textcoords="offset points", ha="right", fontsize=8.5, color=INK2)

ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = sorted(g["bsz"].unique())
ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("batch size (prompts per step)", color=INK2, fontsize=10)
ax.set_ylabel("steps to 50% AIME 1983-2024 (interpolated)", color=INK2, fontsize=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("GRPO on-policy, n = 16 rollouts, no downsampling: KL coef 1e-3 vs 1e-2",
             loc="left", fontsize=11.5, color=INK, pad=10)

sm = plt.cm.ScalarMappable(norm=lrnorm, cmap=CMAP); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("learning rate of each run", color=INK2, fontsize=9.5)
cb.ax.tick_params(colors=INK2, labelsize=8.5, length=0); cb.outline.set_visible(False)

legend = [
    Line2D([], [], marker="o", ls="-", lw=2, ms=9, mfc=SURFACE, mec=INK, mew=2, color=INK,
           label="KL coef 1e-3: fastest LR per batch (circles)"),
    Line2D([], [], marker="D", ls="--", lw=2, ms=7.5, mfc=SURFACE, mec=INK2, mew=2, color=INK2,
           label="KL coef 1e-2: fastest LR per batch (diamonds)"),
    Line2D([], [], marker="o", ls="", ms=6, mfc=RAMP[3], mec=SURFACE,
           label="every run that reached 50%, colored by LR"),
    Line2D([], [], ls=":", lw=1.3, color=MUTED, label="perfect 1/batch scaling from batch 4"),
    Line2D([], [], marker=".", ls="", ms=7, color=INK, label="centre dot = fixed loss scaling (after 2026-09-11)"),
]
ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8.8, labelcolor=INK2)
fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)

print("random picks among duplicates containing a _v2 run (seed", seed, "):")
for c in choices:
    print(f"  kl={c[0]:g} bsz={c[1]} lr={c[2]:g}: {c[4]} steps={c[5]} out of {c[3]} runs")
for kl, b in bests.items():
    print(f"\nKL {kl:g} fastest per batch:\n", b[["bsz", "lr", "steps", "run"]].to_string(index=False))
print("wrote", png_out)
