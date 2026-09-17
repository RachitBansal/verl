"""Reward curves: downsample (64 generated -> K trained) vs plain GRPO with n = K rollouts.

One column per K in {2, 4, 8, 16}; both arms use batch 128 prompts, so both train on
128*K sequences per step. Top row: training reward (critic/score/mean), raw per-step values
faint and a centered rolling mean bold. Bottom row: AIME 1983-2024 mean@1 (every 25 steps).

  orange  downsample N=64->K, fixed scaling, lr 1e-5      (the new runs)
  blue    plain GRPO n=K, lr 1e-5                          (matched LR; pre-fix scaling)
  light blue, dashed  plain GRPO n=K, lr 3e-6              (the plain sweep's best LR for small n)
  blue, dotted        plain GRPO n=16, lr 1e-5, fixed scaling (K = 16 only)
Duplicate runs of one config (seeds, resumes) are drawn as thinner lines of the same colour.

Usage: python plot_curves_downsample_vs_n.py curves.csv out.png [smooth_window]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
ORANGE, BLUE, BLUE_LIGHT = "#eb6834", "#2a78d6", "#86b6ef"
KS = [2, 4, 8, 16]

csv_in, png_out = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 25
df = pd.read_csv(csv_in)

STYLE = {  # (role, lr) -> colour, linestyle, label
    ("downsample", 1e-5):  dict(color=ORANGE,     ls="-",  label="downsample 64→K, fixed scaling, lr 1e-5"),
    ("plain", 1e-5):       dict(color=BLUE,       ls="-",  label="plain GRPO n=K, lr 1e-5"),
    ("plain", 3e-6):       dict(color=BLUE_LIGHT, ls="--", label="plain GRPO n=K, lr 3e-6"),
    ("plain_fixed", 1e-5): dict(color=BLUE,       ls=":",  label="plain GRPO n=K, lr 1e-5, fixed scaling"),
}


def style_for(role, lr):
    for (ro, l), s in STYLE.items():
        if ro == role and abs(lr - l) / l < 0.05:
            return s
    return None


fig, axes = plt.subplots(2, len(KS), figsize=(4.0 * len(KS), 6.6), facecolor=SURFACE, sharey="row")
used = {}
for j, k in enumerate(KS):
    sub = df[df["k"] == k]
    ax_t, ax_v = axes[0, j], axes[1, j]
    for ax in (ax_t, ax_v):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax_t.set_title(f"K = {k}   ({128 * k:,} sequences / step)", loc="left", fontsize=10.5, color=INK, pad=8)

    for (role, lr), grp in sub.groupby(["role", "lr"]):
        s = style_for(role, lr)
        if s is None:
            continue
        runs = sorted(grp["run"].unique(), key=lambda n: -len(grp[grp["run"] == n]))   # longest first
        for i, run in enumerate(runs):
            g = grp[grp["run"] == run].sort_values("step")
            lw = 2.0 if i == 0 else 1.1
            alpha = 1.0 if i == 0 else 0.75
            t = g.dropna(subset=["train"])
            ax_t.plot(t["step"], t["train"], color=s["color"], lw=0.6, alpha=0.18, zorder=2)
            sm = t["train"].rolling(W, center=True, min_periods=max(3, W // 3)).mean()
            ax_t.plot(t["step"], sm, color=s["color"], ls=s["ls"], lw=lw, alpha=alpha, zorder=4)
            v = g.dropna(subset=["val"]).sort_values("step")
            ax_v.plot(v["step"], v["val"], color=s["color"], ls=s["ls"], lw=lw, alpha=alpha,
                      marker="o", ms=3 if i == 0 else 2, mec=SURFACE, mew=0.5, zorder=4)
            used[(role, round(lr, 12))] = s
        if len(runs) > 1:
            ax_t.annotate(f"{len(runs)} runs", (0.98, 0.04), xycoords="axes fraction", ha="right",
                          fontsize=7.5, color=s["color"], zorder=6)

    ax_v.set_xlabel("training step", color=INK2, fontsize=9)
    ax_v.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)

axes[0, 0].set_ylabel(f"train reward ({W}-step mean)", color=INK2, fontsize=9)
axes[1, 0].set_ylabel("AIME accuracy (mean@1)", color=INK2, fontsize=9)
axes[1, 0].annotate("50%", (0, 0.5), xytext=(3, 2), textcoords="offset points", fontsize=7.5, color=INK2)

fig.suptitle("Same sequences per step, different baseline: 64-rollout advantages downsampled to K  vs  plain GRPO with K rollouts\n"
             "batch 128 prompts, KL 1e-3.  Plain lr 1e-5 runs are pre-fix (effective LR ≈ 0.86–0.99× nominal at these n); downsample runs on fixed scaling.",
             x=0.01, ha="left", fontsize=10.5, color=INK)
seen, handles = set(), []
for s in STYLE.values():            # legend in STYLE order, only for series that were actually drawn
    if s["label"] in seen or s not in used.values():
        continue
    seen.add(s["label"])
    handles.append(Line2D([], [], color=s["color"], ls=s["ls"], lw=2, label=s["label"]))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.8,
           labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("wrote", png_out)
