"""Plot steps-to-50% AIME accuracy over the (batch size, learning rate) grid.

Input: CSV produced by pull_steps_to_50.py (one row per wandb run).
Runs with identical (n, bsz, lr) are collapsed to the fastest one.

Usage: python plot_steps_to_50.py steps_to_50.csv out.png [pct] [--by-n]
  --by-n: x-axis = rollouts per prompt n at batch 128 (the rollout sweep) instead of batch size at n = 16
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
# dark = fewest steps (best), light = most steps
CMAP = LinearSegmentedColormap.from_list("blue_rev", RAMP[::-1])
MISSING = "#a8a7a2"
SHORT_STEPS = 200  # runs with fewer validated steps than this are "too early to tell"

BY_N = "--by-n" in sys.argv
args = [a for a in sys.argv[1:] if a != "--by-n"]
csv_in, png_out = args[0], args[1]
PCT = int(args[2]) if len(args) > 2 else 50   # accuracy threshold in percent
df = pd.read_csv(csv_in)
# interpolated crossing when the pull provides it (breaks 25-step ties), else first checkpoint
COL = f"steps_to_{PCT}_interp" if f"steps_to_{PCT}_interp" in df else f"steps_to_{PCT}"
df = df.dropna(subset=["bsz", "lr"])
if "kl" in df:
    df = df[df["kl"].round(6) == 1e-3]
if BY_N:
    df = df[df["bsz"] == 128]   # rollout sweep: every n at batch 128
else:
    df = df[df["n"] == 16]   # batch sweep: n = 16 rollouts per prompt only
# runs on the fixed dp_actor loss normalisation (2026-09-11) carry _updated_scaling in the name
df["fixed"] = df["name"].str.contains("updated_scaling", na=False)

# collapse duplicates: fastest run wins; keep the count and the longest run
df = df.sort_values(COL, na_position="last")
g = df.groupby(["n", "bsz", "lr"], as_index=False).agg(
    steps=(COL, "min"),
    n_runs=("id", "count"),
    longest=("last_val_step", "max"),
    max_val=("max_val", "max"),
    fixed=("fixed", "first"),   # whether the fastest run in the group is post-fix
)
g["n"] = g["n"].astype(int); g["bsz"] = g["bsz"].astype(int)
g.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

hit = g.dropna(subset=["steps"])
norm = LogNorm(vmin=hit["steps"].min(), vmax=hit["steps"].max())


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel(ax, sub, xcol, xlabel):
    sub = sub.copy()
    ok = sub.dropna(subset=["steps"])
    miss = sub[sub["steps"].isna()]
    short = miss[miss["longest"].fillna(0) < SHORT_STEPS]   # not enough training yet
    miss = miss[miss["longest"].fillna(0) >= SHORT_STEPS]

    # never reached 50 %: hollow gray; too short to judge: gray x
    ax.scatter(miss[xcol], miss["lr"], s=70, facecolors="none", edgecolors=MISSING,
               linewidths=1.4, zorder=2)
    ax.scatter(short[xcol], short["lr"], s=60, marker="x", c=MISSING, linewidths=1.4, zorder=2)
    # reached: filled, color = steps (dark = fewer)
    ax.scatter(ok[xcol], ok["lr"], s=90, c=ok["steps"], cmap=CMAP, norm=norm,
               edgecolors=SURFACE, linewidths=1.5, zorder=3)
    # post-fix runs (any status): small ink dot at the centre
    fx = sub[sub["fixed"].fillna(False).astype(bool)]
    ax.scatter(fx[xcol], fx["lr"], s=11, c=INK, zorder=6.5)

    # per-x best: ring + label; connect with dashed line
    best = ok.loc[ok.groupby(xcol)["steps"].idxmin()].sort_values(xcol)
    ax.plot(best[xcol], best["lr"], ls="--", lw=1.2, color=INK2, zorder=2.5)
    ax.scatter(best[xcol], best["lr"], s=260, facecolors="none", edgecolors=INK,
               linewidths=1.8, zorder=4)
    # labels alternate above/below the ring so neighbours never collide
    for i, (_, r) in enumerate(best.iterrows()):
        dy = 11 if i % 2 == 0 else -17
        ax.annotate(f"{r['steps']:.0f}", (r[xcol], r["lr"]), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK, zorder=5)

    # global best on this panel: star (all ties)
    gb = ok[ok["steps"] == ok["steps"].min()]
    ax.scatter(gb[xcol], gb["lr"], marker="*", s=420, c=gb["steps"], cmap=CMAP,
               norm=norm, edgecolors=INK, linewidths=1.2, zorder=6)

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    xs = sorted(sub[xcol].unique())
    ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel("learning rate", color=INK2, fontsize=10)
    style(ax)
    return best, gb


fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURFACE)

if BY_N:
    best_a, gb_a = panel(ax, g, "n", "rollouts per prompt (n), batch 128 prompts")
    ax.set_title("batch 128, no downsampling. Pre-fix runs at n = 32 / 64 stepped at 0.69× / 0.46× their nominal LR (Adam ε regime)",
                 loc="left", fontsize=10.5, color=INK, pad=10)
else:
    best_a, gb_a = panel(ax, g, "bsz", "batch size (prompts per step)")
    ax.set_title("n = 16 rollouts per prompt, no downsampling", loc="left",
                 fontsize=11.5, color=INK, pad=10)

sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cb.set_label(f"steps to {PCT}% AIME 1983-2024, interpolated (darker = fewer)",
             color=INK2, fontsize=9.5)
cb.ax.tick_params(colors=INK2, labelsize=8.5, length=0)
cb.outline.set_visible(False)
cb.ax.invert_yaxis()

legend = [
    Line2D([], [], marker="o", ls="", ms=8, mfc=RAMP[4], mec=SURFACE, label=f"reached {PCT}%"),
    Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=MISSING, mew=1.4,
           label=f"never reached {PCT}%"),
    Line2D([], [], marker="x", ls="", ms=7, color=MISSING, mew=1.4,
           label=f"< {SHORT_STEPS} steps logged (too early)"),
    Line2D([], [], marker="o", ls="--", ms=11, mfc="none", mec=INK, mew=1.6, color=INK2,
           label="fastest LR at each x (steps labeled)"),
    Line2D([], [], marker="*", ls="", ms=14, mfc=RAMP[6], mec=INK, label="fastest overall"),
    Line2D([], [], marker=".", ls="", ms=7, color=INK, label="centre dot = fixed loss scaling (runs after 2026-09-11)"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8.8,
           labelcolor=INK2, bbox_to_anchor=(0.45, -0.07))
fig.suptitle(f"GRPO on-policy, KL coef 1e-3: how fast each ({'rollout count' if BY_N else 'batch size'}, LR) hits {PCT}% val accuracy",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.0)
fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)

xc = "n" if BY_N else "bsz"
print(f"per-{xc} fastest:\n", best_a[[xc, "lr", "steps", "n_runs"]].to_string(index=False))
print("overall fastest:\n", gb_a[[xc, "lr", "steps"]].to_string(index=False))
print("wrote", png_out)
