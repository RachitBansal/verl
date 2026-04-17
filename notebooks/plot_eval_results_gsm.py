"""
GSM8K evaluation results plotting script.
Reproduces Plot 3 from plot_eval_results.ipynb:
  Combined plot of Pretrain, SFT (rgsm interleave), and RL-only.

Usage:
    python notebooks/plot_eval_results_gsm.py
"""

from pathlib import Path
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIRS = [
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results"),
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results_sunny"),
]
DATASET_KEY = "test_score/openai/gsm8k"
TOKEN_MULTIPLIER = 2_000_000

# Plot 3 settings
PLOT_SAMPLES = [1, 8, 32]
TARGET_TEMP = 0.6
TARGET_SHOT = 8       # for base model
RL_ROLLOUTS = 32


# ─── read_score ───────────────────────────────────────────────────────────────
def read_score(result_path: Path, samples: int, dataset_key: str = DATASET_KEY):
    """Read pass@k scores from majority-vote text or dict-style logs."""
    if not result_path.exists():
        return {}

    text = result_path.read_text().splitlines()

    # Majority-format: lines like "  Pass@1 : 0.0403 (4.03%)"
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

    # Dict-style log: scan from bottom for dict containing the key
    for line in reversed(text):
        try:
            payload = ast.literal_eval(line.strip())
        except Exception:
            continue
        if isinstance(payload, dict) and dataset_key in payload:
            return {samples: payload[dataset_key]}
    return {}


# ─── Collect results ──────────────────────────────────────────────────────────

# 1. Base pretrain: "1B-step{step}-{shot}shot-{samples}samples-temp{temp}"
pretrain_rows = []
pre_pattern = re.compile(
    r"1B-step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 2. Direct RL: "olmo2_1b_step{pt_step}_omigsm8k_n{rollouts}[-v{seed}]-step{rl_step}-rl-..."
rl_rows = []
rl_patterns = [
    re.compile(r"olmo2_1b_step(?P<pt_step>\d+)_omigsm8k_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
    re.compile(r"olmo2_1b_step(?P<pt_step>\d+)_omigsm8k_n(?P<num_rollouts>\d+)_v(?P<seed>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
]

# 3. Interleave SFT-Multi (rgsm = repeated SFT data, multiple solutions per question):
#    "OLMo2-1B_step{pt_step}_interleave_twoloader_n32_sft_50000_ppo_0_rgsm-step{rl_step}-rl-..."
interleave_sft_multi_rows = []
interleave_sft_multi_pattern = re.compile(
    r"OLMo2-1B_step(?P<pt_step>\d+)_interleave_twoloader_n(?P<num_rollouts>\d+)_sft_(?P<sft_steps>\d+)_ppo_(?P<ppo_flag>\d+)_rgsm-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 4. Interleave SFT-Single (gsm = non-repeated SFT data, one solution per question):
#    "OLMo2-1B_step{pt_step}_interleave_twoloader_n32_sft_50000_ppo_0_gsm-step{rl_step}-rl-..."
interleave_sft_single_rows = []
interleave_sft_single_pattern = re.compile(
    r"OLMo2-1B_step(?P<pt_step>\d+)_interleave_twoloader_n(?P<num_rollouts>\d+)_sft_(?P<sft_steps>\d+)_ppo_(?P<ppo_flag>\d+)_gsm-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

# 5. SFT->RL (sfted interleave runs): base model is first SFT'd, then RL'd
#    "OLMo2-1B_step{pt_step}sfted_interleave_twoloader_n32_sft_0_ppo_50000_rgsm-step{rl_step}-rl-..."
sftrl_rows = []
sftrl_pattern = re.compile(
    r"OLMo2-1B_step(?P<pt_step>\d+)sfted_interleave_twoloader_n(?P<num_rollouts>\d+)_sft_0_ppo_50000_rgsm-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)

for BASE_DIR in BASE_DIRS:
    if not BASE_DIR.exists():
        continue
    for path in BASE_DIR.iterdir():
        if not path.is_dir():
            continue
        name = path.name

        # Base pretrain
        pre_match = pre_pattern.match(name)
        if pre_match and not any(tag in name for tag in ["-rl-", "-sft-", "-hf"]):
            samples = int(pre_match.group("samples"))
            result_file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                pretrain_rows.append({
                    "step": int(pre_match.group("step")),
                    "shot": int(pre_match.group("shot")),
                    "samples": k,
                    "temp": float(pre_match.group("temp")),
                    "score": score,
                })
            continue

        # Interleave SFT-Multi (rgsm) — check before RL and single patterns
        isft_multi_match = interleave_sft_multi_pattern.match(name)
        if isft_multi_match:
            samples = int(isft_multi_match.group("samples"))
            result_file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                interleave_sft_multi_rows.append({
                    "pt_step": int(isft_multi_match.group("pt_step")),
                    "rl_step": int(isft_multi_match.group("rl_step")),
                    "samples": k,
                    "temp": float(isft_multi_match.group("temp")),
                    "num_rollouts": int(isft_multi_match.group("num_rollouts")),
                    "score": score,
                })
            continue

        # SFT->RL (sfted interleave) — check before RL and single patterns
        sftrl_match = sftrl_pattern.match(name)
        if sftrl_match:
            samples = int(sftrl_match.group("samples"))
            result_file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                sftrl_rows.append({
                    "pt_step": int(sftrl_match.group("pt_step")),
                    "rl_step": int(sftrl_match.group("rl_step")),
                    "samples": k,
                    "temp": float(sftrl_match.group("temp")),
                    "num_rollouts": int(sftrl_match.group("num_rollouts")),
                    "score": score,
                })
            continue

        # Interleave SFT-Single (gsm) — check before RL patterns
        isft_single_match = interleave_sft_single_pattern.match(name)
        if isft_single_match:
            samples = int(isft_single_match.group("samples"))
            result_file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
            for k, score in read_score(path / result_file, samples=samples).items():
                interleave_sft_single_rows.append({
                    "pt_step": int(isft_single_match.group("pt_step")),
                    "rl_step": int(isft_single_match.group("rl_step")),
                    "samples": k,
                    "temp": float(isft_single_match.group("temp")),
                    "num_rollouts": int(isft_single_match.group("num_rollouts")),
                    "score": score,
                })
            continue

        # Direct RL
        for pat in rl_patterns:
            rl_match = pat.match(name)
            if rl_match:
                samples = int(rl_match.group("samples"))
                result_file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
                seed = int(rl_match.group("seed")) if "seed" in rl_match.groupdict() and rl_match.group("seed") else 1
                for k, score in read_score(path / result_file, samples=samples).items():
                    rl_rows.append({
                        "pt_step": int(rl_match.group("pt_step")),
                        "rl_step": int(rl_match.group("rl_step")),
                        "samples": k,
                        "temp": float(rl_match.group("temp")),
                        "num_rollouts": int(rl_match.group("num_rollouts")),
                        "seed": seed,
                        "score": score,
                    })
                break

# 5. One-off experiments (hardcoded paths)
ONE_OFF_EXPERIMENTS = {
    "interleave_20_1": {
        "path": "OLMo2-1B_step1000_interleave_twoloader_n32_sft_20_ppo_1_eos-step15000-rl-0shot-boxed-32samples-temp0.6",
        "pt_step": 1000,
    },
    "combined_1_1": {
        "path": "OLMo2-1B_step1000_combined_twoloader_n32_rl1.0_sft1.0_eos-step1880-rl-0shot-boxed-32samples-temp0.6",
        "pt_step": 1000,
    },
    "interleave_20_1_step7000": {
        "path": "OLMo2-1B_step7000_interleave_twoloader_n32_sft_20_ppo_1_rgsm-step9000-rl-0shot-boxed-32samples-temp0.6",
        "pt_step": 7000,
    },
    "combined_1_1_step10000": {
        "path": "OLMo2-1B_step10000_combined_twoloader_n32_rl1.0_sft1.0_eos-step820-rl-0shot-boxed-32samples-temp0.6",
        "pt_step": 10000,
    },
    "curriculum_step1000": {
        "path": "OLMo2-1B_step1000_curriculum_sft200_ppo1_rlinc1_sftdec1-step4750-rl-0shot-boxed-32samples-temp0.6",
        "pt_step": 1000,
    },
}

# Pre-computed pass@k scores from parquet predictions (scored offline)
PRECOMPUTED_SCORES = {
    "combined_1_1_step10000": {1: 0.4778, 8: 0.7435, 32: 0.8529},
    "interleave_20_1_step7000": {1: 0.4645, 8: 0.7178, 32: 0.8317},
    "curriculum_step1000": {1: 0.2476, 8: 0.4997, 32: 0.6596},
}
oneoff_scores = {}  # key -> {k: score}
for key, info in ONE_OFF_EXPERIMENTS.items():
    for BASE_DIR in BASE_DIRS:
        result_path = BASE_DIR / info["path"] / "gsm8k_majority_results.txt"
        scores = read_score(result_path, samples=32)
        if scores:
            oneoff_scores[key] = scores
            break
    if key not in oneoff_scores and key in PRECOMPUTED_SCORES:
        oneoff_scores[key] = PRECOMPUTED_SCORES[key]
    if key not in oneoff_scores:
        print(f"Warning: no results found for {key}")

pre_df = pd.DataFrame(pretrain_rows)
rl_df = pd.DataFrame(rl_rows)
interleave_sft_multi_df = pd.DataFrame(interleave_sft_multi_rows)
interleave_sft_single_df = pd.DataFrame(interleave_sft_single_rows)
sftrl_df = pd.DataFrame(sftrl_rows)

# Deduplicate
if not pre_df.empty:
    pre_df = pre_df.drop_duplicates(subset=["step", "shot", "samples", "temp"], keep="first")
if not rl_df.empty:
    rl_df = rl_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts", "seed"], keep="first")
if not interleave_sft_multi_df.empty:
    interleave_sft_multi_df = interleave_sft_multi_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts"], keep="first")
if not interleave_sft_single_df.empty:
    interleave_sft_single_df = interleave_sft_single_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts"], keep="first")
if not sftrl_df.empty:
    sftrl_df = sftrl_df.drop_duplicates(subset=["pt_step", "rl_step", "samples", "temp", "num_rollouts"], keep="first")

print(f"Loaded: {len(pre_df)} pretrain, {len(rl_df)} RL, {len(interleave_sft_multi_df)} SFT-Multi (rgsm), {len(interleave_sft_single_df)} SFT-Single (gsm), {len(sftrl_df)} SFT->RL (sfted)")


# ─── Plot 3: Base + Direct RL + SFT (rgsm) ───────────────────────────────────

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
    "legend.fontsize": 25,
})

color_base = "#777777"
color_rl = "#E24A33"
color_sft_multi = "#009E73"    # Emerald green (repeated/multi-solution SFT)
color_sft_single = "#56B4E9"   # Light blue (single-solution SFT)
color_sftrl = "#7B3294"        # Purple (SFT->RL)
color_interleave = "#CC79A7"   # Muted pink (interleave 20:1)
color_combined = "#D55E00"     # Vermillion orange (combined 1:1)
color_curriculum = "#0072B2"   # Dark blue (curriculum)

styles = {
    "pretrain": {
        "color": color_base, "marker": "o", "ls": ":", "markersize": 10,
        "linewidth": 2.5, "label": r"$\mathcal{M}_t^{\text{SFT-Multi}}$"[:-1] + "PLACEHOLDER",  # placeholder
    },
    "sft_multi": {
        "color": color_sft_multi, "marker": "d", "ls": "-.", "markersize": 12,
        "linewidth": 2.5, "label": r"$\mathcal{M}_t^{\text{SFT-Multi}}$",
    },
    "sft_single": {
        "color": color_sft_single, "marker": "d", "ls": "--", "markersize": 12,
        "linewidth": 2.5, "label": r"$\mathcal{M}_t^{\text{SFT-Single}}$",
    },
    "rl": {
        "color": color_rl, "marker": "*", "ls": "-", "markersize": 18,
        "linewidth": 3.5, "label": r"$\mathcal{M}_t^{\text{RL}}$",
    },
    "sftrl": {
        "color": color_sftrl, "marker": "D", "ls": "-", "markersize": 10,
        "linewidth": 3.5, "label": r"$\mathcal{M}_t^{\text{SFT}\to\text{RL}}$",
    },
}
# Fix pretrain style separately (avoid the placeholder hack)
styles["pretrain"] = {
    "color": color_base, "marker": "o", "ls": ":", "markersize": 10,
    "linewidth": 2.5, "label": r"$\mathcal{M}_t$ (Base)",
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

    # Base model
    pre_curve = pre_df[
        (pre_df["samples"] == samples)
        & (pre_df["shot"] == TARGET_SHOT)
        & (pre_df["temp"] == TARGET_TEMP)
    ].sort_values("step")
    if not pre_curve.empty:
        ax.plot(pre_curve["step"] * TOKEN_MULTIPLIER, pre_curve["score"] * 100, **styles["pretrain"])

    # SFT-Multi (rgsm interleave) — use last rl_step per pt_step
    sft_multi_curve = interleave_sft_multi_df[
        (interleave_sft_multi_df["samples"] == samples)
        & (interleave_sft_multi_df["temp"] == TARGET_TEMP)
        & (interleave_sft_multi_df["num_rollouts"] == RL_ROLLOUTS)
    ]
    if not sft_multi_curve.empty:
        sft_multi_last = sft_multi_curve.loc[sft_multi_curve.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
        ax.plot(sft_multi_last["pt_step"] * TOKEN_MULTIPLIER, sft_multi_last["score"] * 100, **styles["sft_multi"])

    # SFT-Single (gsm interleave) — use last rl_step per pt_step
    sft_single_curve = interleave_sft_single_df[
        (interleave_sft_single_df["samples"] == samples)
        & (interleave_sft_single_df["temp"] == TARGET_TEMP)
        & (interleave_sft_single_df["num_rollouts"] == RL_ROLLOUTS)
    ]
    if not sft_single_curve.empty:
        sft_single_last = sft_single_curve.loc[sft_single_curve.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
        ax.plot(sft_single_last["pt_step"] * TOKEN_MULTIPLIER, sft_single_last["score"] * 100, **styles["sft_single"])

    # SFT->RL — use last rl_step per pt_step
    if not sftrl_df.empty:
        sftrl_curve = sftrl_df[
            (sftrl_df["samples"] == samples)
            & (sftrl_df["temp"] == TARGET_TEMP)
        ]
        if not sftrl_curve.empty:
            sftrl_last = sftrl_curve.loc[sftrl_curve.groupby("pt_step")["rl_step"].idxmax()].sort_values("pt_step")
            ax.plot(sftrl_last["pt_step"] * TOKEN_MULTIPLIER, sftrl_last["score"] * 100, **styles["sftrl"], zorder=10)

    # Direct RL — use last rl_step per pt_step, best seed
    rl_subset = rl_df[
        (rl_df["samples"] == samples)
        & (rl_df["temp"] == TARGET_TEMP)
        & (rl_df["num_rollouts"] == RL_ROLLOUTS)
    ]
    if not rl_subset.empty:
        rl_last = rl_subset.loc[rl_subset.groupby(["pt_step", "seed"])["rl_step"].idxmax()].sort_values("pt_step")
        rl_last = rl_last.loc[rl_last.groupby("pt_step")["score"].idxmax()]
        ax.plot(rl_last["pt_step"] * TOKEN_MULTIPLIER, rl_last["score"] * 100, **styles["rl"], zorder=10)

    # One-off dots
    for key, pt_step, color, marker, label_text in [
        ("interleave_20_1", 1000, color_interleave, "P", r"$\mathcal{M}_t^{\text{Interleave-20:1}}$"),
        ("combined_1_1", 1000, color_combined, "X", r"$\mathcal{M}_t^{\text{Combined-1:1}}$"),
        ("interleave_20_1_step7000", 7000, color_interleave, "P", None),
        ("combined_1_1_step10000", 10000, color_combined, "X", None),
        ("curriculum_step1000", 1000, color_curriculum, "^", r"$\mathcal{M}_t^{\text{Curriculum}}$"),
    ]:
        if key in oneoff_scores and samples in oneoff_scores[key]:
            ax.scatter(pt_step * TOKEN_MULTIPLIER, oneoff_scores[key][samples] * 100,
                       color=color, marker=marker, s=200, zorder=15, edgecolors="k", linewidths=0.5,
                       label=label_text if idx == 0 else None)

    # Formatting
    ax.set_title(f"Pass@{samples}", pad=15)
    ax.set_xlabel("Pre-training tokens (steps)")
    ax.xaxis.set_major_formatter(formatter)
    if idx == 0:
        ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", color="gray", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

# Legend — collect from all axes to catch series that may only appear in some panels
handles, labels = [], []
for a in axes:
    h, l = a.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in labels:
            handles.append(hi)
            labels.append(li)
desired_order = [
    styles["pretrain"]["label"],
    styles["rl"]["label"],
    styles["sftrl"]["label"],
    styles["sft_multi"]["label"],
    styles["sft_single"]["label"],
    r"$\mathcal{M}_t^{\text{Interleave-20:1}}$",
    r"$\mathcal{M}_t^{\text{Combined-1:1}}$",
    r"$\mathcal{M}_t^{\text{Curriculum}}$",
]
order_lookup = {label: i for i, label in enumerate(desired_order)}
sorted_pairs = sorted(zip(handles, labels), key=lambda pair: order_lookup.get(pair[1], 99))
sorted_handles, sorted_labels = zip(*sorted_pairs)

plt.tight_layout()
plt.subplots_adjust(bottom=0.38)
fig.legend(
    sorted_handles, sorted_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.15),
    ncol=3, frameon=True, framealpha=1.0, borderpad=0.3,
)

output_path = Path(__file__).parent / "gsm_passatk_comparison.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.show()
