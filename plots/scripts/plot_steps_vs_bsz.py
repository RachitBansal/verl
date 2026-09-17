"""Steps to reach 50% AIME accuracy as a function of batch size (KL coef 1e-3, n = 16).

For each batch size, the headline curve is the fastest learning rate tried.
Every (batch, LR) that reached 50% is also shown as a small point colored by LR.
Duplicate runs of one config: if a "_v2" run is among them, one run is drawn at
random (seeded); otherwise the fastest run is kept. Runs with fewer than
SHORT_STEPS validated steps are excluded from the draw.

Usage: python plot_steps_vs_bsz.py steps_to_50_kl.csv out.png [seed]
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

csv_in, png_out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed)

df = pd.read_csv(csv_in).dropna(subset=["bsz", "lr"])
df = df[(df["kl"].round(6) == 1e-3) & (df["n"] == 16)]
# short runs are excluded only while they have not reached 50% (a hit is a hit)
df = df[(df["last_val_step"].fillna(0) >= SHORT_STEPS) | df["steps_to_50"].notna()]
df["bsz"] = df["bsz"].astype(int)

# ---- collapse duplicates -----------------------------------------------------
rows, choices = [], []
for (bsz, lr), s in df.groupby(["bsz", "lr"]):
    if len(s) > 1 and s["name"].str.contains("_v2").any():
        pick = s.iloc[rng.integers(len(s))]
        choices.append((bsz, lr, len(s), pick["name"], pick["id"], pick["steps_to_50"]))
    elif len(s) > 1:
        pick = s.sort_values("steps_to_50", na_position="last").iloc[0]
    else:
        pick = s.iloc[0]
    rows.append(dict(bsz=bsz, lr=lr, steps=pick["steps_to_50"], run=pick["name"], id=pick["id"],
                     longest=pick["last_val_step"], n_candidates=len(s),
                     fixed="updated_scaling" in str(pick["name"])))
g = pd.DataFrame(rows)
g.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

hit = g.dropna(subset=["steps"])
best = hit.loc[hit.groupby("bsz")["steps"].idxmin()].sort_values("bsz")
never = g[g["steps"].isna()].groupby("bsz").size()

# ---- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

lrnorm = LogNorm(vmin=hit["lr"].min(), vmax=hit["lr"].max())
ax.scatter(hit["bsz"], hit["steps"], s=42, c=hit["lr"], cmap=CMAP, norm=lrnorm,
           edgecolors=SURFACE, linewidths=1.0, zorder=3)
# post-fix runs: small ink dot at the centre (drawn last so it shows on the headline markers too)
fx = hit[hit["fixed"]]
ax.scatter(fx["bsz"], fx["steps"], s=10, c=INK, zorder=7)

# perfect scaling: steps ∝ 1/bsz, anchored at the smallest batch's best point
b0, s0 = best["bsz"].iloc[0], best["steps"].iloc[0]
xs = np.array([best["bsz"].min() / 1.3, best["bsz"].max() * 1.3])
ax.plot(xs, s0 * b0 / xs, ls=":", lw=1.3, color=MUTED, zorder=2)
ax.annotate("perfect scaling\n(steps ∝ 1/batch)", (xs[1], s0 * b0 / xs[1]), xytext=(-4, 8),
            textcoords="offset points", ha="right", fontsize=8.5, color=INK2)

# headline curve: fastest LR per batch
ax.plot(best["bsz"], best["steps"], lw=2, color=INK, zorder=4)
ax.scatter(best["bsz"], best["steps"], s=120, facecolors=SURFACE, edgecolors=INK, linewidths=2,
           zorder=5)
for i, (_, r) in enumerate(best.iterrows()):
    below = i == len(best) - 2   # second-to-last label goes under the point to avoid the last one
    ax.annotate(f"{int(r['steps'])} steps\nlr {r['lr']:g}", (r["bsz"], r["steps"]),
                xytext=(10, -28) if below else (10, 6),
                textcoords="offset points", fontsize=8.2, color=INK, zorder=6)

ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = sorted(g["bsz"].unique())
ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("batch size (prompts per step)", color=INK2, fontsize=10)
ax.set_ylabel("training steps to reach 50% AIME 1983-2024", color=INK2, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("GRPO on-policy, KL coef 1e-3, n = 16 rollouts, no downsampling", loc="left",
             fontsize=11.5, color=INK, pad=10)

sm = plt.cm.ScalarMappable(norm=lrnorm, cmap=CMAP); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("learning rate of each run", color=INK2, fontsize=9.5)
cb.ax.tick_params(colors=INK2, labelsize=8.5, length=0); cb.outline.set_visible(False)

legend = [
    Line2D([], [], marker="o", ls="-", lw=2, ms=9, mfc=SURFACE, mec=INK, mew=2, color=INK,
           label="fastest LR at each batch size"),
    Line2D([], [], marker="o", ls="", ms=6, mfc=RAMP[3], mec=SURFACE, label="every (batch, LR) run that reached 50%"),
    Line2D([], [], ls=":", lw=1.3, color=MUTED, label="perfect 1/batch scaling from the smallest batch"),
    Line2D([], [], marker=".", ls="", ms=7, color=INK, label="centre dot = fixed loss scaling (after 2026-09-11)"),
]
ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8.8, labelcolor=INK2)
fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)

print("random picks among duplicates containing a _v2 run (seed", seed, "):")
for c in choices:
    print(f"  bsz={c[0]} lr={c[1]:g}: {c[3]} ({c[4]}) steps={c[5]} out of {c[2]} runs")
print("\nfastest LR per batch:\n", best[["bsz", "lr", "steps", "run"]].to_string(index=False))
print("\nconfigs that never reached 50% per batch:\n", never.to_string())
print("wrote", png_out)
