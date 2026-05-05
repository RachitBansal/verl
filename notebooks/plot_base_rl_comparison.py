"""
Compare two base models (50B vs 60B-MATH-heavy) and their direct RL results on MATH.

Series:
  1. M_t (50B Base)       — gray dotted circles
  2. M_t^RL (50B RL)      — red solid stars
  3. M_t (60B Base)        — blue dotted circles
  4. M_t^RL (60B RL)       — orange solid stars

Usage:
    python notebooks/plot_base_rl_comparison.py
"""

from pathlib import Path
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

# --- Config -----------------------------------------------------------------
BASE_DIRS = [
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results"),
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results_sunny"),
]
DATASET_KEY = "test_score/DigitalLearningGmbH/MATH-lighteval"
TOKEN_MULTIPLIER = 2_000_000

PLOT_SAMPLES = [1, 8, 32]
TARGET_TEMP = 0.6
TARGET_SHOT = 8


# --- read_score -------------------------------------------------------------
def read_score(result_path: Path, samples: int, dataset_key: str = DATASET_KEY):
    """Read pass@k scores from majority-vote text or dict-style logs."""
    if not result_path.exists():
        return {}

    text = result_path.read_text().splitlines()

    if samples > 1 or "majority" in result_path.name:
        scores = {}
        for line in text:
            if "Pass@" in line and ":" in line:
                k_match = re.search(r"Pass@(\d+)", line)
                if k_match:
                    k = int(k_match.group(1))
                    parts = line.split(":")
                    if len(parts) > 1:
                        try:
                            score = float(parts[1].split()[0])
                            scores[k] = score
                        except Exception:
                            pass
        if scores:
            return scores

    for line in reversed(text):
        try:
            payload = ast.literal_eval(line.strip())
        except Exception:
            continue
        if isinstance(payload, dict) and dataset_key in payload:
            return {samples: payload[dataset_key]}
    return {}


# --- Patterns ---------------------------------------------------------------

# 50B Base: "1B-step..." or "1B-stage1-50B-step..."
pre50_pattern = re.compile(
    r"1B-(?:stage1-50B-)?step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 50B Direct RL: "olmo2_1b_step{pt_step}_omi_n{rollouts}-step{rl_step}-rl-..."
rl50_pattern = re.compile(
    r"olmo2_1b_step(?P<pt_step>\d+)_omi_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 60B Base: "1B-MATH60B-step{step}-{shot}shot-{samples}samples-temp{temp}"
pre60_pattern = re.compile(
    r"1B-MATH60B-step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 60B Direct RL: "olmo2_1b_60bmath_step{pt_step}_omi_n{rollouts}-step{rl_step}-rl-..."
rl60_pattern = re.compile(
    r"olmo2_1b_60bmath_step(?P<pt_step>\d+)_omi_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 4B Base (0-shot boxed only — 8-shot evals failed): "4B-step{step}-0shot-boxed-{samples}samples-temp{temp}"
# NOTE: 4B base uses 0-shot (not 8-shot like 1B), results are not directly comparable
pre4b_pattern = re.compile(
    r"4B-step(?P<step>\d+)-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 4B Direct RL: "olmo2_4b_step{pt_step}_omi_n{rollouts}-step{rl_step}-rl-..."
rl4b_pattern = re.compile(
    r"olmo2_4b_step(?P<pt_step>\d+)_omi_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)


# --- Collect results --------------------------------------------------------
pre50_rows, rl50_rows = [], []
pre60_rows, rl60_rows = [], []
pre4b_rows, rl4b_rows = [], []

for BASE_DIR in BASE_DIRS:
    if not BASE_DIR.exists():
        continue
    for path in BASE_DIR.iterdir():
        if not path.is_dir():
            continue
        name = path.name


        # 50B Base
        m = pre50_pattern.match(name)
        if m and not any(tag in name for tag in ["-rl-", "-sft-", "-hf"]):
            samples = int(m.group("samples"))
            result_file = "math_majority_results.txt" if samples > 1 else "math_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                pre50_rows.append({
                    "step": int(m.group("step")),
                    "shot": int(m.group("shot")),
                    "samples": k,
                    "temp": float(m.group("temp")),
                    "score": score,
                })
            continue

        # 50B RL
        m = rl50_pattern.match(name)
        if m:
            samples = int(m.group("samples"))
            result_file = "math_majority_results.txt" if samples > 1 else "math_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                rl50_rows.append({
                    "pt_step": int(m.group("pt_step")),
                    "rl_step": int(m.group("rl_step")),
                    "samples": k,
                    "temp": float(m.group("temp")),
                    "num_rollouts": int(m.group("num_rollouts")),
                    "score": score,
                })

        # 60B Base (check before 50B to avoid partial match)
        m = pre60_pattern.match(name)
        if m:
            samples = int(m.group("samples"))
            result_file = "math_majority_results.txt" if samples > 1 else "math_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                pre60_rows.append({
                    "step": int(m.group("step")),
                    "shot": int(m.group("shot")),
                    "samples": k,
                    "temp": float(m.group("temp")),
                    "score": score,
                })
            continue

        # 60B RL (check before 50B RL)
        m = rl60_pattern.match(name)
        if m:
            samples = int(m.group("samples"))
            result_file = "math_majority_results.txt" if samples > 1 else "math_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                rl60_rows.append({
                    "pt_step": int(m.group("pt_step")),
                    "rl_step": int(m.group("rl_step")),
                    "samples": k,
                    "temp": float(m.group("temp")),
                    "num_rollouts": int(m.group("num_rollouts")),
                    "score": score,
                })
            continue

        # 4B Base (0-shot boxed)
        m = pre4b_pattern.match(name)
        if m:
            samples = int(m.group("samples"))
            result_file = "math_majority_results.txt" if samples > 1 else "math_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                pre4b_rows.append({
                    "step": int(m.group("step")),
                    "samples": k,
                    "temp": float(m.group("temp")),
                    "score": score,
                })
            continue

        # 4B RL
        m = rl4b_pattern.match(name)
        if m:
            samples = int(m.group("samples"))
            result_file = "math_majority_results.txt" if samples > 1 else "math_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                rl4b_rows.append({
                    "pt_step": int(m.group("pt_step")),
                    "rl_step": int(m.group("rl_step")),
                    "samples": k,
                    "temp": float(m.group("temp")),
                    "num_rollouts": int(m.group("num_rollouts")),
                    "score": score,
                })
            continue

        

pre50_df = pd.DataFrame(pre50_rows)
rl50_df = pd.DataFrame(rl50_rows)
pre60_df = pd.DataFrame(pre60_rows)
rl60_df = pd.DataFrame(rl60_rows)
pre4b_df = pd.DataFrame(pre4b_rows)
rl4b_df = pd.DataFrame(rl4b_rows)

# Deduplicate
if not pre50_df.empty:
    pre50_df = pre50_df.drop_duplicates(subset=["step", "shot", "samples", "temp"], keep="first")
if not rl50_df.empty:
    rl50_df = rl50_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts"], keep="first")
if not pre60_df.empty:
    pre60_df = pre60_df.drop_duplicates(subset=["step", "shot", "samples", "temp"], keep="first")
if not rl60_df.empty:
    rl60_df = rl60_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts"], keep="first")
if not pre4b_df.empty:
    pre4b_df = pre4b_df.drop_duplicates(subset=["step", "samples", "temp"], keep="first")
if not rl4b_df.empty:
    rl4b_df = rl4b_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts"], keep="first")

print(f"Loaded: {len(pre50_df)} 50B-base, {len(rl50_df)} 50B-RL, "
      f"{len(pre60_df)} 60B-base, {len(rl60_df)} 60B-RL, "
      f"{len(pre4b_df)} 4B-base, {len(rl4b_df)} 4B-RL")


# --- Plot -------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.fontsize": 22,
})

# Color pairs: base (lighter) and RL (darker) share the same hue family
# Red family: 1B-50B
color_base50 = "#F4A582"     # Light salmon
color_rl50 = "#B2182B"       # Dark red
# Blue family: 1B-60B-MATH
color_base60 = "#92C5DE"     # Light blue
color_rl60 = "#2166AC"       # Dark blue
# Green family: 4B
color_base4b = "#A6D96A"     # Light green
color_rl4b = "#1A9641"       # Dark green

styles = {
    "base50": {
        "color": color_base50, "marker": "o", "ls": ":", "markersize": 10,
        "linewidth": 2.5, "label": r"$\mathcal{M}_t$ (1B-50B Base)",
    },
    "rl50": {
        "color": color_rl50, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": r"$\mathcal{M}_t^{\text{RL}}$ (1B-50B)",
    },
    "base60": {
        "color": color_base60, "marker": "o", "ls": ":", "markersize": 10,
        "linewidth": 2.5, "label": r"$\mathcal{M}_t$ (1B-60B-MATH Base)",
    },
    "rl60": {
        "color": color_rl60, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": r"$\mathcal{M}_t^{\text{RL}}$ (1B-60B-MATH)",
    },
    "base4b": {
        "color": color_base4b, "marker": "o", "ls": ":", "markersize": 10,
        "linewidth": 2.5, "label": r"$\mathcal{M}_t$ (4B Base, 0-shot$^\dagger$)",
    },
    "rl4b": {
        "color": color_rl4b, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": r"$\mathcal{M}_t^{\text{RL}}$ (4B)",
    },
}

def dual_format(num, _pos):
    tok = num
    mag = 0
    while abs(tok) >= 1000:
        mag += 1
        tok /= 1000.0
    tok_str = "%.0f%s" % (tok, ["", "K", "M", "B", "T"][mag])
    # step = num / TOKEN_MULTIPLIER
    # smag = 0
    # while abs(step) >= 1000:
    #     smag += 1
    #     step /= 1000.0
    # step_str = "%.0f%s" % (step, ["", "k", "m", "b"][smag])
    return tok_str  # f"{tok_str}\n{step_str}"

formatter = FuncFormatter(dual_format)

fig, axes = plt.subplots(1, 4, figsize=(23, 6), sharey=False, constrained_layout=False,
                         gridspec_kw={"width_ratios": [1, 1, 1, 0.55]})
for ax in axes[1:3]:
    ax.sharey(axes[0])
    ax.tick_params(labelleft=False)

diff_styles = {
    "50B": {
        "color": color_rl50, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": "1B Params - 50B Tokens",
    },
    "60B": {
        "color": color_rl60, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": "1B Params - 60B Tokens",
    },
    "4B": {
        "color": color_rl4b, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": "4B Params - 50B Tokens",
    },
}

all_diffs = []

for idx, samples in enumerate(PLOT_SAMPLES):
    ax = axes[idx]

    # 50B: RL − Base
    rl_last_50, base50_at = None, None
    if not rl50_df.empty:
        rl_sub = rl50_df[
            (rl50_df["samples"] == samples)
            & (rl50_df["temp"] == TARGET_TEMP)
            & (rl50_df["num_rollouts"] == 64)
        ]
        if not rl_sub.empty:
            rl_last_50 = rl_sub.loc[rl_sub.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
    if not pre50_df.empty:
        curve = pre50_df[
            (pre50_df["samples"] == samples)
            & (pre50_df["shot"] == TARGET_SHOT)
            & (pre50_df["temp"] == TARGET_TEMP)
        ].sort_values("step")
        if not curve.empty:
            _bx, _by = curve["step"].values, curve["score"].values * 100
            base50_at = lambda x, bx=_bx, by=_by: np.interp(x, bx, by)
    if rl_last_50 is not None and base50_at is not None:
        dx = rl_last_50["pt_step"].values
        dy = rl_last_50["score"].values * 100 - base50_at(dx)
        all_diffs.extend(dy.tolist())
        ax.plot(dx * TOKEN_MULTIPLIER, dy, **diff_styles["50B"], zorder=10)

    # 60B: RL − Base
    rl_last_60, base60_at = None, None
    if not pre60_df.empty:
        curve60 = pre60_df[
            (pre60_df["samples"] == samples)
            & (pre60_df["shot"] == TARGET_SHOT)
            & (pre60_df["temp"] == TARGET_TEMP)
        ].sort_values("step")
        if not curve60.empty:
            _bx, _by = curve60["step"].values, curve60["score"].values * 100
            base60_at = lambda x, bx=_bx, by=_by: np.interp(x, bx, by)
    if not rl60_df.empty:
        rl60_sub = rl60_df[
            (rl60_df["samples"] == samples)
            & (rl60_df["temp"] == TARGET_TEMP)
        ]
        if not rl60_sub.empty:
            rl_last_60 = rl60_sub.loc[rl60_sub.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
    if rl_last_60 is not None and base60_at is not None:
        dx = rl_last_60["pt_step"].values
        dy = rl_last_60["score"].values * 100 - base60_at(dx)
        all_diffs.extend(dy.tolist())
        ax.plot(dx * TOKEN_MULTIPLIER, dy, **diff_styles["60B"], zorder=10)

    # 4B: RL − Base
    rl_last_4b, base4b_at = None, None
    if not pre4b_df.empty:
        curve4b = pre4b_df[
            (pre4b_df["samples"] == samples)
            & (pre4b_df["temp"] == TARGET_TEMP)
        ].sort_values("step")
        if not curve4b.empty:
            _bx, _by = curve4b["step"].values, curve4b["score"].values * 100
            base4b_at = lambda x, bx=_bx, by=_by: np.interp(x, bx, by)
    if not rl4b_df.empty:
        rl4b_sub = rl4b_df[
            (rl4b_df["samples"] == samples)
            & (rl4b_df["temp"] == TARGET_TEMP)
        ]
        if not rl4b_sub.empty:
            rl_last_4b = rl4b_sub.loc[rl4b_sub.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
    if rl_last_4b is not None and base4b_at is not None:
        dx = rl_last_4b["pt_step"].values
        dy = rl_last_4b["score"].values * 100 - base4b_at(dx)
        all_diffs.extend(dy.tolist())
        ax.plot(dx * TOKEN_MULTIPLIER, dy, **diff_styles["4B"], zorder=10)

    # Formatting
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Pass@{samples}", pad=15)
    ax.set_xlabel("Pre-training tokens")
    ax.xaxis.set_major_locator(MultipleLocator(10_000_000_000))
    ax.xaxis.set_major_formatter(formatter)
    if idx == 0:
        ax.set_ylabel("RL − Base Acc (pp)", labelpad=12)
    ax.grid(True, linestyle=":", color="gray", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

# Set ylim dynamically so the top tick is visible with headroom
if all_diffs:
    y_top = max(all_diffs)
    axes[0].set_ylim(top=y_top * 1.15)

# Model description panel (right)
from matplotlib.lines import Line2D

ax_desc = axes[3]
ax_desc.set_axis_off()
model_handles = [
    Line2D([0, 1], [0, 0], color=diff_styles["50B"]["color"], marker=diff_styles["50B"]["marker"],
           ls=diff_styles["50B"]["ls"], markersize=14, linewidth=2.5),
    Line2D([0, 1], [0, 0], color=diff_styles["60B"]["color"], marker=diff_styles["60B"]["marker"],
           ls=diff_styles["60B"]["ls"], markersize=14, linewidth=2.5),
    Line2D([0, 1], [0, 0], color=diff_styles["4B"]["color"], marker=diff_styles["4B"]["marker"],
           ls=diff_styles["4B"]["ls"], markersize=14, linewidth=2.5),
]
model_labels = [
    "N = 1B \nD = 50B (Mix A)",
    "N = 1B \nD = 60B (Mix A $\cup$ Mix B)",
    "N = 4B \nD = 50B (Mix A)",
]
ax_desc.legend(
    model_handles, model_labels,
    loc="center left", bbox_to_anchor=(-0.15, 0.5), frameon=True, framealpha=1.0,
    fontsize=18, title="Models and Pretraining Data", title_fontsize=20,
    handlelength=2.5, labelspacing=1.2, borderpad=0.8,
)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(left=0.07)

output_path = Path(__file__).parent / "math_base_rl_comparison.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
png_path = output_path.with_suffix(".png")
plt.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"Saved to {png_path}")
plt.show()
