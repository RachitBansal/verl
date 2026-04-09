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

# 50B Base: "1B-step{step}-{shot}shot-{samples}samples-temp{temp}"
pre50_pattern = re.compile(
    r"1B-step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
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

def dual_format(num, pos):
    tok = num
    mag = 0
    while abs(tok) >= 1000:
        mag += 1
        tok /= 1000.0
    tok_str = "%.0f%s" % (tok, ["", "K", "M", "B", "T"][mag])
    step = num / TOKEN_MULTIPLIER
    smag = 0
    while abs(step) >= 1000:
        smag += 1
        step /= 1000.0
    step_str = "%.0f%s" % (step, ["", "k", "m", "b"][smag])
    return f"{tok_str}\n{step_str}"

formatter = FuncFormatter(dual_format)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, samples in enumerate(PLOT_SAMPLES):
    ax = axes[idx]

    # 50B Base
    curve = pre50_df[
        (pre50_df["samples"] == samples)
        & (pre50_df["shot"] == TARGET_SHOT)
        & (pre50_df["temp"] == TARGET_TEMP)
    ].sort_values("step")
    if not curve.empty:
        ax.plot(curve["step"] * TOKEN_MULTIPLIER, curve["score"] * 100, **styles["base50"])

    # 50B RL — best rl_step per pt_step, use n=64
    rl_sub = rl50_df[
        (rl50_df["samples"] == samples)
        & (rl50_df["temp"] == TARGET_TEMP)
        & (rl50_df["num_rollouts"] == 64)
    ]
    if not rl_sub.empty:
        rl_last = rl_sub.loc[rl_sub.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
        ax.plot(rl_last["pt_step"] * TOKEN_MULTIPLIER, rl_last["score"] * 100, **styles["rl50"], zorder=10)

    # 60B Base
    curve60 = pre60_df[
        (pre60_df["samples"] == samples)
        & (pre60_df["shot"] == TARGET_SHOT)
        & (pre60_df["temp"] == TARGET_TEMP)
    ].sort_values("step")
    if not curve60.empty:
        ax.plot(curve60["step"] * TOKEN_MULTIPLIER, curve60["score"] * 100, **styles["base60"])

    # 60B RL — best rl_step per pt_step (n=32)
    rl60_sub = rl60_df[
        (rl60_df["samples"] == samples)
        & (rl60_df["temp"] == TARGET_TEMP)
    ]
    if not rl60_sub.empty:
        rl60_last = rl60_sub.loc[rl60_sub.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
        ax.plot(rl60_last["pt_step"] * TOKEN_MULTIPLIER, rl60_last["score"] * 100, **styles["rl60"], zorder=10)

    # 4B Base (0-shot boxed — no shot filter needed, pattern already selects 0-shot)
    curve4b = pre4b_df[
        (pre4b_df["samples"] == samples)
        & (pre4b_df["temp"] == TARGET_TEMP)
    ].sort_values("step")
    if not curve4b.empty:
        ax.plot(curve4b["step"] * TOKEN_MULTIPLIER, curve4b["score"] * 100, **styles["base4b"])

    # 4B RL — best rl_step per pt_step
    rl4b_sub = rl4b_df[
        (rl4b_df["samples"] == samples)
        & (rl4b_df["temp"] == TARGET_TEMP)
    ]
    if not rl4b_sub.empty:
        rl4b_last = rl4b_sub.loc[rl4b_sub.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
        ax.plot(rl4b_last["pt_step"] * TOKEN_MULTIPLIER, rl4b_last["score"] * 100, **styles["rl4b"], zorder=10)

    # Formatting
    ax.set_title(f"Pass@{samples}", pad=15)
    ax.set_xlabel("Pre-training tokens (steps)")
    ax.xaxis.set_major_locator(MultipleLocator(10_000_000_000))  # every 10B tokens
    ax.xaxis.set_major_formatter(formatter)
    if idx == 0:
        ax.set_ylabel("MATH Accuracy (%)")
    ax.set_ylim(-5, 80)
    ax.grid(True, linestyle=":", color="gray", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
desired_order = [
    styles["base50"]["label"],
    styles["rl50"]["label"],
    styles["base60"]["label"],
    styles["rl60"]["label"],
    styles["base4b"]["label"],
    styles["rl4b"]["label"],
]
order_lookup = {label: i for i, label in enumerate(desired_order)}
sorted_pairs = sorted(zip(handles, labels), key=lambda pair: order_lookup.get(pair[1], 99))
sorted_handles, sorted_labels = zip(*sorted_pairs)

plt.tight_layout()
plt.subplots_adjust(bottom=0.45)
fig.legend(
    sorted_handles, sorted_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.12),
    ncol=3, frameon=True, framealpha=1.0, borderpad=0.3,
)

# Footnote about 4B base using 0-shot
fig.text(0.5, -0.28, r"$\dagger$ 4B Base uses 0-shot evaluation (8-shot evals not yet available)",
         ha="center", fontsize=16, fontstyle="italic", color="#555555")

output_path = Path(__file__).parent / "math_base_rl_comparison.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.show()
