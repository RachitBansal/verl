"""Steps-to-50% over the (batch size, LR) grid, comparing KL coef 1e-3 vs 1e-2.

Marker shape encodes the KL coefficient (circle = 1e-3, diamond = 1e-2);
color encodes steps to 50% exactly as in plot_steps_to_50.py. n = 16 runs only.

Usage: python plot_steps_to_50_kl.py steps_to_50_kl.csv out.png
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
CMAP = LinearSegmentedColormap.from_list("blue_rev", RAMP[::-1])  # dark = fewest steps
MISSING = "#a8a7a2"
SHORT_STEPS = 200

# per-KL styling: marker, x-nudge (log2 units) so co-located points stay visible, line style
KL_STYLE = {
    1e-3: dict(marker="o", nudge=-0.10, ls="--", label="KL coef 1e-3", s=90),
    1e-2: dict(marker="D", nudge=+0.10, ls=":",  label="KL coef 1e-2", s=70),
}

csv_in, png_out = sys.argv[1], sys.argv[2]
PCT = int(sys.argv[3]) if len(sys.argv) > 3 else 50   # accuracy threshold in percent
df = pd.read_csv(csv_in).dropna(subset=["bsz", "lr"])
# interpolated crossing when the pull provides it (breaks 25-step ties), else first checkpoint
COL = f"steps_to_{PCT}_interp" if f"steps_to_{PCT}_interp" in df else f"steps_to_{PCT}"
df = df[df["n"] == 16]
df["kl"] = df["kl"].round(6)
# runs on the fixed dp_actor loss normalisation (2026-09-11) carry _updated_scaling in the name
df["fixed"] = df["name"].str.contains("updated_scaling", na=False)

df = df.sort_values(COL, na_position="last")
g = df.groupby(["kl", "bsz", "lr"], as_index=False).agg(
    steps=(COL, "min"), n_runs=("id", "count"),
    longest=("last_val_step", "max"), max_val=("max_val", "max"),
    fixed=("fixed", "first"))   # whether the fastest run in the group is post-fix
g["bsz"] = g["bsz"].astype(int)
g.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

hit = g.dropna(subset=["steps"])
norm = LogNorm(vmin=hit["steps"].min(), vmax=hit["steps"].max())

fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

bests = {}
for kl, st in KL_STYLE.items():
    sub = g[np.isclose(g["kl"], kl)].copy()
    if sub.empty:
        continue
    x = sub["bsz"] * 2.0 ** st["nudge"]
    ok = sub["steps"].notna()
    short = ~ok & (sub["longest"].fillna(0) < SHORT_STEPS)
    miss = ~ok & ~short

    ax.scatter(x[miss], sub.loc[miss, "lr"], s=st["s"] * 0.8, marker=st["marker"],
               facecolors="none", edgecolors=MISSING, linewidths=1.4, zorder=2)
    ax.scatter(x[short], sub.loc[short, "lr"], s=st["s"] * 0.7, marker="x", c=MISSING,
               linewidths=1.4, zorder=2)
    ax.scatter(x[ok], sub.loc[ok, "lr"], s=st["s"], marker=st["marker"], c=sub.loc[ok, "steps"],
               cmap=CMAP, norm=norm, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    # post-fix runs (any status): small ink dot at the centre
    fx = sub["fixed"].fillna(False).astype(bool)
    ax.scatter(x[fx], sub.loc[fx, "lr"], s=11, c=INK, zorder=6.5)

    okdf = sub[ok].assign(x=x[ok])
    best = okdf.loc[okdf.groupby("bsz")["steps"].idxmin()].sort_values("bsz")
    bests[kl] = best
    ax.plot(best["x"], best["lr"], ls=st["ls"], lw=1.3, color=INK2, zorder=2.5)
    ax.scatter(best["x"], best["lr"], s=st["s"] * 3.0, marker=st["marker"], facecolors="none",
               edgecolors=INK, linewidths=1.7, zorder=4)
    for i, (_, r) in enumerate(best.iterrows()):
        # 1e-3 labels sit above-left, 1e-2 labels below-right, so the two series never collide
        dx, dy = (-6, 12) if kl == 1e-3 else (6, -18)
        ax.annotate(f"{r['steps']:.0f}", (r["x"], r["lr"]), xytext=(dx, dy),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK, zorder=5)

# overall fastest across both KLs: star(s)
gb = hit[hit["steps"] == hit["steps"].min()]
for _, r in gb.iterrows():
    st = KL_STYLE[min(KL_STYLE, key=lambda k: abs(k - r["kl"]))]
    ax.scatter([r["bsz"] * 2.0 ** st["nudge"]], [r["lr"]], marker="*", s=440, c=[r["steps"]],
               cmap=CMAP, norm=norm, edgecolors=INK, linewidths=1.2, zorder=6)

ax.set_xscale("log", base=2); ax.set_yscale("log")
xs = sorted(g["bsz"].unique())
ax.set_xticks(xs); ax.set_xticklabels([str(int(v)) for v in xs])
ax.set_xlabel("batch size (prompts per step)", color=INK2, fontsize=10)
ax.set_ylabel("learning rate", color=INK2, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("n = 16 rollouts per prompt, no downsampling", loc="left", fontsize=11.5,
             color=INK, pad=10)

sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cb.set_label(f"steps to {PCT}% AIME 1983-2024, interpolated (darker = fewer)", color=INK2, fontsize=9.5)
cb.ax.tick_params(colors=INK2, labelsize=8.5, length=0)
cb.outline.set_visible(False); cb.ax.invert_yaxis()

legend = [
    Line2D([], [], marker="o", ls="--", ms=9, mfc=RAMP[4], mec=SURFACE, color=INK2,
           label="KL coef 1e-3 (circles; dashed = fastest LR per batch)"),
    Line2D([], [], marker="D", ls=":", ms=7.5, mfc=RAMP[4], mec=SURFACE, color=INK2,
           label="KL coef 1e-2 (diamonds; dotted = fastest LR per batch)"),
    Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=MISSING, mew=1.4, label=f"never reached {PCT}%"),
    Line2D([], [], marker="x", ls="", ms=7, color=MISSING, mew=1.4, label=f"< {SHORT_STEPS} steps logged"),
    Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=INK, mew=1.6, label="fastest at that batch (steps labeled)"),
    Line2D([], [], marker="*", ls="", ms=14, mfc=RAMP[6], mec=INK, label="fastest overall"),
    Line2D([], [], marker=".", ls="", ms=7, color=INK, label="centre dot = fixed loss scaling (runs after 2026-09-11)"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8.8,
           labelcolor=INK2, bbox_to_anchor=(0.45, -0.08))
fig.suptitle(f"GRPO on-policy: steps to {PCT}% val accuracy across (batch size, LR), KL coef 1e-3 vs 1e-2",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.0)
fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)

for kl, b in bests.items():
    print(f"KL {kl:g} fastest per batch:\n", b[["bsz", "lr", "steps", "n_runs"]].to_string(index=False))
print("overall fastest:\n", gb[["kl", "bsz", "lr", "steps"]].to_string(index=False))
print("wrote", png_out)
