"""Paper figure: how the best learning rate moves when the step is grown two ways.

  left  - batch size B varied at n = 16 rollouts (KL coef 1e-3 circles, 1e-2 diamonds)
  right - rollouts per prompt n varied at B = 128 (KL coef 1e-3)

Each marker is a (B or n, LR) configuration, coloured by interpolated steps to 50% AIME (shared
colour scale, darker = fewer). Hollow grey markers never reached 50%. The ringed marker in each
column is the fastest LR at that B or n, labelled with its step count, and the rings are joined.
The fastest run per configuration is used regardless of code version, except that at n = 32/64 only
the fixed-loss-scaling reruns are eligible (the earlier runs there sat in Adam's eps regime).
A run that never crossed is drawn hollow if it finished and ran at least as long as the fastest crossing in its
column (so early-stopped brackets at large B count); shorter or still-running non-crossers are omitted. Writes the PNG next to the other figures and the PDF to plots/paper/.

Usage: python plot_lr_scaling_paper.py csv/steps_to_50_kl.csv png/lr_scaling_paper.png
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue_rev", RAMP[::-1])          # dark = fewest steps
SHORT_STEPS = 200
FIXED_ONLY_N = {32, 64, 128}
KL_STYLE = {1e-3: dict(marker="o", nudge=-0.10, ls="--", s=95, lab_off=(-9, 9), lab_ha="right"),
            1e-2: dict(marker="D", nudge=+0.10, ls=":", s=75, lab_off=(9, -14), lab_ha="left")}

csv_in, png_out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_in).dropna(subset=["bsz", "lr"])
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df = df[~df["name"].str.startswith("downsample")].copy()
df["fixed"] = df["name"].str.contains("updated_scaling", na=False)
df["kl"] = df["kl"].astype(float).round(6)
# A run that never crossed counts as "did not reach 50%" only if it is finished and ran at least as long as the
# fastest crossing at its (kl, bsz) or (n) column, so the big-batch brackets we stopped early (they had already
# been beaten) show up as hollow markers while a run killed after a handful of steps does not. Columns with no
# crossing at all fall back to SHORT_STEPS.
df["last_val_step"] = df["last_val_step"].fillna(0)
best_b = df[df["n"] == 16].groupby(["kl", "bsz"])[COL].min().rename("best_b")
best_n = df[df["bsz"] == 128].groupby(["kl", "n"])[COL].min().rename("best_n")
df = df.join(best_b, on=["kl", "bsz"]).join(best_n, on=["kl", "n"])
ref = df[["best_b", "best_n"]].min(axis=1).fillna(SHORT_STEPS)
long_enough = (df["last_val_step"] >= np.minimum(ref, SHORT_STEPS)) & (df["state"] != "running")
df = df[df[COL].notna() | long_enough].drop(columns=["best_b", "best_n"])


def collapse(src, keys):
    """fastest run per configuration (ties: the longest run)."""
    src = src.sort_values([COL, "last_val_step"], na_position="last", ascending=[True, False])
    g = src.groupby(keys, as_index=False).first()
    return g.rename(columns={COL: "steps"})


# left: (kl, bsz, lr) at n = 16;  right: (n, lr) at bsz 128, KL 1e-3, n = 32/64 fixed-code only
left = collapse(df[df["n"] == 16], ["kl", "bsz", "lr"])
left = left[left["kl"].isin([1e-3, 1e-2])]
rsrc = df[(df["bsz"] == 128) & np.isclose(df["kl"], 1e-3) & (~df["n"].isin(FIXED_ONLY_N) | df["fixed"])]
right = collapse(rsrc, ["n", "lr"])

pd.concat([left.assign(panel="batch_sweep"), right.assign(panel="rollout_sweep", kl=1e-3)]).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

hits = pd.concat([left.dropna(subset=["steps"])["steps"], right.dropna(subset=["steps"])["steps"]])
norm = LogNorm(vmin=hits.min(), vmax=hits.max())

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 13, "legend.fontsize": 11.5})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, facecolor=SURFACE,
                         gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.06))


def draw(ax, data, xcol, kl_values, xlabel):
    ax.set_facecolor(SURFACE)
    for kl in kl_values:
        st = KL_STYLE[kl]
        sub = data[np.isclose(data["kl"], kl)] if "kl" in data and len(kl_values) > 1 else data
        x = sub[xcol] * 2.0 ** (st["nudge"] if len(kl_values) > 1 else 0.0)
        miss = sub["steps"].isna()
        ax.scatter(x[miss], sub.loc[miss, "lr"], s=st["s"] * 0.8, marker=st["marker"], facecolors="none",
                   edgecolors=MUTED, linewidths=1.4, zorder=3)
        ax.scatter(x[~miss], sub.loc[~miss, "lr"], s=st["s"], marker=st["marker"], c=sub.loc[~miss, "steps"],
                   cmap=CMAP, norm=norm, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        ok = sub.dropna(subset=["steps"])
        best = ok.loc[ok.groupby(xcol)["steps"].idxmin()].sort_values(xcol)
        bx = best[xcol] * 2.0 ** (st["nudge"] if len(kl_values) > 1 else 0.0)
        ax.plot(bx, best["lr"], ls=st["ls"], lw=1.4, color=INK2, zorder=2.5)
        ax.scatter(bx, best["lr"], s=st["s"] * 3.0, marker=st["marker"], facecolors="none", edgecolors=INK,
                   linewidths=1.7, zorder=5)
        for xv, (_, r) in zip(bx, best.iterrows()):
            ax.annotate(f"{r['steps']:.0f}", (xv, r["lr"]), xytext=st["lab_off"], textcoords="offset points",
                        ha=st["lab_ha"], fontsize=9.5, color=INK, zorder=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    xs = sorted(data[xcol].unique())
    ax.set_xticks(xs); ax.set_xticklabels([str(int(v)) for v in xs])
    ax.set_xlim(min(xs) / 1.7, max(xs) * 1.7)
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.7)
    ax.tick_params(which="both", colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


draw(axes[0], left, "bsz", [1e-3, 1e-2], "batch size B  (prompts per step, n = 16 rollouts)")
draw(axes[1], right, "n", [1e-3], "rollouts per prompt n  (B = 128 prompts)")
axes[0].set_ylabel("learning rate", color=INK)
axes[0].set_title("more prompts", loc="left", fontsize=13, color=INK)
axes[1].set_title("more rollouts", loc="left", fontsize=13, color=INK)

sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.03)
cb.set_label("steps to 50% AIME 1983–2024 (darker = fewer)", color=INK2)
cb.ax.tick_params(colors=INK2, length=0)
cb.outline.set_visible(False)

legend = [
    Line2D([], [], marker="o", ls="--", ms=9, mfc=RAMP[4], mec=SURFACE, color=INK2, label="KL coef 1e-3; dashed line joins the fastest LR at each B or n"),
    Line2D([], [], marker="D", ls=":", ms=7.5, mfc=RAMP[4], mec=SURFACE, color=INK2, label="KL coef 1e-2; dotted line joins the fastest LR at each B"),
    Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=MUTED, mew=1.4, label="did not reach 50% (ran at least as long as the fastest LR in its column)"),
    Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=INK, mew=1.7, label="fastest LR at that B or n, steps to 50% labelled"),
]
fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.47, -0.005), columnspacing=3.0)
fig.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.25)
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper")   # plots/paper/
os.makedirs(pdf_dir, exist_ok=True)
pdf_out = os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf"))
fig.savefig(pdf_out, facecolor=SURFACE)
print("wrote", png_out, "and", pdf_out)
for name, d, xcol in [("batch sweep", left, "bsz"), ("rollout sweep", right, "n")]:
    ok = d.dropna(subset=["steps"])
    b = ok.loc[ok.groupby([c for c in ("kl", xcol) if c in ok])["steps"].idxmin()]
    print(f"\n{name}: fastest LR per column\n", b[[c for c in ("kl", xcol, "lr", "steps", "name") if c in b]].to_string(index=False))
