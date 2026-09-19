"""Paper figure: steps to 50% AIME vs sequences trained per step, for two ways of growing the step.

  batch sweep   - plain GRPO, n = 16 rollouts per prompt, batch size B varied (fastest LR at each B)
  rollout sweep - plain GRPO, B = 128 prompts, rollouts per prompt n varied (fastest LR at each n)

Both series use KL coef 1e-3 and the fastest available run per point (pre-fix or fixed loss scaling;
for n = 32/64 only the fixed-code reruns are eligible because the pre-fix runs sat in Adam's eps regime).
No pre/post-fix comparison, no downsampling series. Writes PNG and PDF.

Usage: python plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_paper.png
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE, VIOLET = "#2a78d6", "#4a3aa7"
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}          # n where a pre-fix run is an honest measurement
FIXED_ONLY_N = {32, 64}                    # n where only fixed-code reruns count

csv_in, png_out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_in)
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df = df[np.isclose(df["kl"].astype(float), 1e-3)].dropna(subset=["seqs", "lr"]).copy()
df["seqs"] = df["seqs"].astype(int)
df = df[~df["downsample"].astype(bool)]
long_enough = (df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[COL].notna()
df = df[long_enough]


def fastest_per(src, key):
    """fastest run at each value of `key` (any LR); returns hits and, separately, keys with no crossing."""
    rows = []
    for k, s in src.groupby(key):
        s = s.sort_values([COL, "last_val_step"], na_position="last", ascending=[True, False])
        p = s.iloc[0]
        rows.append(dict(key=k, seqs=int(p["seqs"]), lr=p["lr"], steps=p[COL], last=p["last_val_step"],
                         max_val=p["max_val"], run=p["name"]))
    out = pd.DataFrame(rows).sort_values("seqs")
    return out.dropna(subset=["steps"]), out[out["steps"].isna()]


batch_src = df[df["n"] == 16]
batch, _ = fastest_per(batch_src, "bsz")

roll_src = df[(df["bsz"] == 128) & (df["n"].isin(VALID_N_PREFIX) | (df["fixed"].astype(bool) & df["n"].isin(FIXED_ONLY_N)))]
roll, roll_never = fastest_per(roll_src, "n")

table = pd.concat([batch.assign(series="batch_sweep_n16"), roll.assign(series="rollout_sweep_bsz128"),
                   roll_never.assign(series="rollout_sweep_bsz128_never")])
table.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)),
                          os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 13, "legend.fontsize": 11.5})
fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# perfect 1/sequences scaling through the smallest batch point
b0 = batch.iloc[0]
xs = np.array([batch["seqs"].min() / 2.5, max(batch["seqs"].max(), roll["seqs"].max()) * 1.6])
ax.plot(xs, b0["steps"] * b0["seqs"] / xs, ls=":", lw=1.2, color=MUTED, zorder=1)

# rollout sweep (violet)
ax.plot(roll["seqs"], roll["steps"], lw=2, color=VIOLET, zorder=4)
ax.scatter(roll["seqs"], roll["steps"], s=120, marker="p", color=VIOLET, edgecolors=SURFACE, linewidths=1.2, zorder=5)
# batch sweep (blue), drawn on top so the shared (B=128, n=16) point reads as part of both
ax.plot(batch["seqs"], batch["steps"], lw=2, color=BLUE, zorder=6)
ax.scatter(batch["seqs"], batch["steps"], s=90, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=7)

# direct labels: B on the blue series (below-left), n on the violet series (above-right)
for _, r in batch.iterrows():
    ax.annotate(f"B={int(r['key'])}", (r["seqs"], r["steps"]), xytext=(-8, -13), textcoords="offset points",
                ha="right", fontsize=10, color=INK2, zorder=8)
for _, r in roll.iterrows():
    if int(r["key"]) == 16:
        continue
    ax.annotate(f"n={int(r['key'])}", (r["seqs"], r["steps"]), xytext=(9, 7), textcoords="offset points",
                ha="left", fontsize=10, color=VIOLET, zorder=8)
if len(roll_never):   # n with no crossing (n = 1: every prompt's advantages are identically zero) -> footnote, not a marker
    note = ", ".join(f"n = {int(r['key'])} (B = 128) never reaches 50% within {int(r['last']):,} steps" for _, r in roll_never.iterrows())
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=10, color=VIOLET, ha="left", va="bottom")

ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = sorted(set(batch["seqs"]) | set(roll["seqs"]))
ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" if t < 10000 else f"{t // 1024}k" for t in ticks], fontsize=11)
ax.set_xlabel("sequences per optimizer step  (prompts B × rollouts n)", color=INK)
ax.set_ylabel("steps to 50% AIME 1983–2024", color=INK)
ax.grid(True, which="major", color=GRID, lw=0.7)
ax.tick_params(which="both", colors=INK2, length=0)
for sp in ax.spines.values():
    sp.set_visible(False)

legend = [
    Line2D([], [], marker="o", ls="-", lw=2, ms=7, color=BLUE, mec=SURFACE, label="more prompts: batch size B varied, n = 16"),
    Line2D([], [], marker="p", ls="-", lw=2, ms=8, color=VIOLET, mec=SURFACE, label="more rollouts: n varied, B = 128"),
    Line2D([], [], ls=":", lw=1.2, color=MUTED, label="perfect scaling (steps ∝ 1/sequences)"),
]
ax.legend(handles=legend, loc="upper right", frameon=False, labelcolor=INK)
fig.tight_layout()
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
fig.savefig(png_out.replace(".png", ".pdf"), facecolor=SURFACE)
print("wrote", png_out, "and .pdf")
print("\nbatch sweep:\n", batch[["key", "seqs", "lr", "steps", "run"]].to_string(index=False))
print("\nrollout sweep:\n", roll[["key", "seqs", "lr", "steps", "run"]].to_string(index=False))
print("\nrollout sweep, never:\n", roll_never[["key", "seqs", "last", "max_val"]].to_string(index=False))
